# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.inference

LLM provider configuration (Groq/OpenAI) and the request/retry/validate
loop that turns discovered parameter/metric candidates into inferred
semantic metadata.

Rate-limit safety is layered, not left to reactive retries alone (see the
"Token estimation, batch-size safety, and a global tokens-per-minute
budget" section below):
  1. Estimate each request's token cost BEFORE sending it.
  2. Keep every request's estimated cost under --inference-tpm-budget by
     splitting batches further when needed (on top of --inference-batch-size's
     item-count cap).
  3. Rate-limit globally: every request, across ALL infer_*_metadata()
     calls in this process, reserves its estimated cost against one
     shared-per-provider tokens-per-minute ledger, so a burst of
     individually-small requests can't collectively exceed the budget
     either.
The existing per-response fail-fast/backoff handling (_tpm_over_limit(),
_is_hard_payload_limit(), etc.) is still there as a safety net for when a
provider rejects a request anyway -- estimation is a heuristic, not a
guarantee -- but the goal of the three layers above is to make that the
exception, not the normal path.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import TYPE_CHECKING

from openai import APIStatusError, NotFoundError, OpenAI, OpenAIError, RateLimitError

if TYPE_CHECKING:
    from metadata.metrics import MetricCandidate
    from metadata.parameters import ParameterCandidate

from ai.prompts import (
    METRIC_SYSTEM_PROMPT, SYSTEM_PROMPT, build_metric_prompt, build_prompt,
)
from ai.validation import extract_json_array, validate_metadata, validate_metric_metadata

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # NOTE: Groq model availability AND per-model rate limits both vary
        # per account/key. Standard instruct models (openai/gpt-oss-120b,
        # openai/gpt-oss-20b, qwen/qwen3.6-27b, ...) are capped at a low 8K
        # tokens-per-minute on Groq's free/on-demand tier -- easy to blow
        # through with this pipeline's prompts (a full main.cc/problem.hh
        # embedded every request). "groq/compound-mini" gets a much higher
        # 70K TPM on the same tier (its sibling "groq/compound" too) despite
        # being a normal chat.completions model otherwise -- it's an
        # "agentic" system that can autonomously invoke web search/code
        # execution, which occasionally adds a stray sentence around the
        # JSON, but ai.validation's extract_json_array() already strips
        # markdown fences and leading/trailing prose, so that's handled.
        # Override per-run with --model, or check what your own key can
        # access and its rate limits:
        #   curl -s https://api.groq.com/openai/v1/models \
        #     -H "Authorization: Bearer $GROQ_API_KEY" | python3 -m json.tool
        "default_model": "groq/compound-mini",
        "signup_url": "https://console.groq.com/keys",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "signup_url": "https://platform.openai.com/api-keys",
    },
}

DEFAULT_PROVIDER = "groq"

#: Fallback wait (seconds) when a token-rate-limit response doesn't tell us
#: how long to wait. Deliberately longer than a full minute -- these are
#: almost always a *tokens-per-minute* budget (not a burst/request-rate
#: one), so anything shorter risks retrying inside the same still-exhausted
#: window and wasting the attempt.
RATE_LIMIT_BACKOFF_SECONDS = 65.0

# =============================================================================
# Token estimation, batch-size safety, and a global tokens-per-minute budget
# =============================================================================
#
# Three layers, applied in order, all working off the SAME estimate/budget
# so they can't disagree with each other:
#   1. estimate_request_tokens() -- guess a request's cost BEFORE sending it
#      (prompt + system prompt + expected completion), not just react to a
#      provider's rejection after the fact.
#   2. _split_by_token_budget() -- split a batch further (below
#      --inference-batch-size's item-count cap) if its estimated cost alone
#      would exceed the target budget, e.g. a couple of items in an
#      8-item batch happened to carry unusually large cpp_hint/context
#      excerpts.
#   3. _RateLimiter -- a sliding 60s tokens-per-minute ledger SHARED across
#      every request this process makes to a given provider, not just
#      within one batch or one infer_*_metadata() call. Batches 1-2 above
#      only bound the size of a single request; without this, five
#      individually-small batches sent back-to-back could still blow
#      through the account's real rolling TPM window, since nothing
#      tracked how much had already gone out in the last 60 seconds.

#: Rough chars-per-token heuristic for estimating a request's size ahead of
#: sending it. ~4 chars/token is the standard rule of thumb for English/code
#: text with GPT-style BPE tokenizers (Groq's hosted models included).
#: Deliberately a planning estimate, not exact accounting -- only a
#: response's own usage.total_tokens (see _RateLimiter.record_actual()) is
#: exact, and that's only known AFTER the request already went out.
CHARS_PER_TOKEN_ESTIMATE = 4.0

#: Rough per-item completion size estimate -- each returned item is a JSON
#: object with semantic_name/datatype/unit/quantityKind/confidence/
#: explanation, plus ini/key/index. Padded generously on purpose:
#: underestimating completion size is exactly what lets a "should fit"
#: prompt-only estimate blow past a TPM budget once the actual response is
#: counted too (Groq's "Requested" figure in a rejection covers both).
COMPLETION_TOKENS_PER_ITEM_ESTIMATE = 150

#: Default target tokens-per-minute budget for both batch-splitting and the
#: global rate limiter below. Deliberately NOT set to a Groq account's full
#: advertised TPM cap (e.g. 12,000 on a typical free/on-demand key) -- token
#: estimation above is a heuristic, not exact, and other activity on the
#: same key (a concurrent run, manual testing in another terminal) eats
#: into the same rolling window. Leaving ~25-30% headroom under a 12K cap
#: is what keeps estimation error from tipping a "should fit" request into
#: an actual 429/413. Override with --inference-tpm-budget to match your
#: own account's real limit (see console.groq.com/docs/rate-limits, or the
#: models-list command referenced in this module's error messages) minus
#: similar headroom -- don't just set it to the raw advertised number.
DEFAULT_TPM_BUDGET = 9000


def estimate_tokens(text: str) -> int:
    """Conservative chars/4 token estimate -- see CHARS_PER_TOKEN_ESTIMATE."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def estimate_request_tokens(system_prompt: str, prompt: str, item_count: int) -> int:
    """Estimated total tokens (prompt + completion) one inference request
    will cost -- what a provider's TPM budget actually counts against, not
    just the prompt half. Used both to decide whether a batch needs
    splitting further (_split_by_token_budget()) and to reserve budget
    against the global rate limiter (_RateLimiter.reserve()) BEFORE the
    request is sent, not just to explain a rejection after the fact.
    """
    return (
        estimate_tokens(system_prompt)
        + estimate_tokens(prompt)
        + item_count * COMPLETION_TOKENS_PER_ITEM_ESTIMATE
    )


def _split_by_token_budget(
    candidates: list,
    build_prompt_fn,
    system_prompt: str,
    token_budget: int,
    item_label: str,
) -> list[list]:
    """Given a list of candidates that already fits --inference-batch-size's
    item-count cap, further split it if its ESTIMATED token cost (prompt +
    system prompt + expected completion, via estimate_request_tokens())
    still exceeds `token_budget` -- e.g. a couple of items in an otherwise
    normal-sized batch happened to carry unusually large cpp_hint/context
    excerpts. Splits in half repeatedly until each piece fits, or is down
    to a single item that still doesn't (nothing left to split then -- it's
    sent as-is with a warning; only a smaller --inference-tpm-budget's
    sibling --inference-batch-size, a different model, or trimming that
    item's own source context could help further, not more splitting).
    """
    if not candidates:
        return []
    prompt = build_prompt_fn(candidates)
    estimated = estimate_request_tokens(system_prompt, prompt, len(candidates))
    if estimated <= token_budget or len(candidates) == 1:
        if estimated > token_budget:
            print(
                f"warning: a single {item_label} still needs an estimated ~{estimated:,} tokens on its "
                f"own -- over the {token_budget:,}-token target, but there's nothing left to split. "
                f"Sending as-is; if the provider rejects it, a larger --inference-tpm-budget won't help "
                f"(try --model, or check what's making this item's own source context so large).",
                file=sys.stderr,
            )
        return [candidates]
    mid = len(candidates) // 2
    return (
        _split_by_token_budget(candidates[:mid], build_prompt_fn, system_prompt, token_budget, item_label)
        + _split_by_token_budget(candidates[mid:], build_prompt_fn, system_prompt, token_budget, item_label)
    )


class _RateLimiter:
    """Sliding 60-second tokens-per-minute budget shared across every
    inference request this process makes to a given provider -- not just
    within one batch or one infer_*_metadata() call (see this section's
    module-level comment above for why that distinction matters).

    reserve(estimated_tokens) blocks (sleeping) until sending that many
    more tokens would stay within `tpm_budget` given everything already
    sent in the trailing 60 seconds, then records the reservation and
    returns a mutable ledger entry. record_actual() later corrects that
    entry with the provider's real usage.total_tokens (successful
    responses) or 0 (a request the provider rejected outright before
    generating anything, e.g. a 404/429/413 -- it never actually consumed
    completion tokens, so holding the optimistic estimate against future
    reserve() calls would throttle harder than necessary). Without this
    correction the limiter would only ever get more conservative over a
    long run, purely from estimation error accumulating.
    """

    def __init__(self, tpm_budget: int):
        self.tpm_budget = max(1, tpm_budget)
        self._usage: list[list] = []  # each entry: [monotonic_timestamp, tokens]

    def _prune(self) -> None:
        now = time.monotonic()
        self._usage = [entry for entry in self._usage if now - entry[0] < 60.0]

    def reserve(self, estimated_tokens: int, *, verbose: bool = False) -> list:
        while True:
            self._prune()
            used = sum(tokens for _, tokens in self._usage)
            if used + estimated_tokens <= self.tpm_budget or not self._usage:
                # Either it fits, or the ledger is already empty and it
                # still doesn't fit -- waiting can't help that second case
                # (the per-request/TPM-over-limit fail-fast checks in
                # _query_llm_json_array handle a genuinely oversized single
                # request), so let it through rather than looping forever.
                break
            oldest_t = self._usage[0][0]
            wait = max(0.5, 60.0 - (time.monotonic() - oldest_t) + 0.5)
            if verbose:
                print(
                    f"[rate-limiter] {used:,}/{self.tpm_budget:,} tokens used in the last 60s across all "
                    f"requests; this one needs ~{estimated_tokens:,} more -- waiting {wait:.1f}s for the "
                    f"window to clear..."
                )
            time.sleep(wait)
        entry = [time.monotonic(), estimated_tokens]
        self._usage.append(entry)
        return entry

    def record_actual(self, entry: list, actual_tokens: int | None) -> None:
        """Correct a reservation with real usage (or 0 for a request that
        never actually ran) once it's known. No-op if actual_tokens is
        None (provider didn't return usage info -- keep the estimate).
        """
        if actual_tokens is not None:
            entry[1] = actual_tokens


#: One limiter per provider, shared for this process's lifetime -- created
#: lazily on first use so importing this module never needs an API key.
_RATE_LIMITERS: dict[str, _RateLimiter] = {}


def _get_rate_limiter(provider: str, tpm_budget: int) -> _RateLimiter:
    limiter = _RATE_LIMITERS.get(provider)
    if limiter is None or limiter.tpm_budget != max(1, tpm_budget):
        # First use, or --inference-tpm-budget changed since the last call
        # (shouldn't happen mid-run in practice -- it's a fixed CLI arg --
        # but starting a fresh ledger rather than reusing a stale-budget
        # one is the safe response if it ever does).
        limiter = _RateLimiter(tpm_budget)
        _RATE_LIMITERS[provider] = limiter
    return limiter


def get_client(provider: str = DEFAULT_PROVIDER) -> OpenAI:
    """
    Create an OpenAI-compatible client for the given provider.

    Requires the provider's API key env var to be set
    (GROQ_API_KEY for Groq, OPENAI_API_KEY for OpenAI).
    """
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDER_CONFIG)}")

    config = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{config['api_key_env']} is not set in the environment. "
            f"Get a free key at {config['signup_url']}."
        )

    return OpenAI(api_key=api_key, base_url=config["base_url"])



def _is_rate_or_size_limited(exc: OpenAIError) -> bool:
    """True for a 429 (RateLimitError) or Groq's 413 'request too large for
    tokens-per-minute budget' response -- both mean "the provider rejected
    this on token/rate budget grounds", not "something is wrong with the
    request itself". Groq's TPM-exceeded case comes back as a generic
    APIStatusError (413 has no dedicated openai-python exception class), so
    it's detected via the response body's {"error": {"code":
    "rate_limit_exceeded"}} instead of the exception type alone.
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        body = getattr(exc, "body", None)
        if isinstance(body, dict) and body.get("error", {}).get("code") == "rate_limit_exceeded":
            return True
        if exc.status_code == 413:
            return True
    return False


#: Groq's TPM-exceeded message embeds the two numbers that matter, e.g.
#: "...on tokens per minute (TPM): Limit 8000, Requested 9401, please...".
_TPM_LIMIT_REQUESTED_PATTERN = re.compile(r"Limit (\d+),\s*Requested (\d+)", re.IGNORECASE)


def _tpm_over_limit(exc: OpenAIError) -> tuple[int, int] | None:
    """If `exc` is a Groq TPM/size-limit error that embeds "Limit N,
    Requested M" AND the single request alone (M) exceeds the whole
    per-minute budget (N), returns (limit, requested) -- this is NOT a
    transient "too much recent traffic" situation: no amount of waiting for
    the window to reset can ever make this exact request fit, since it
    alone is bigger than the entire budget. Returns None otherwise
    (including when the numbers are present but M <= N -- that case IS
    plausibly transient and the normal backoff-and-retry applies).
    """
    body = getattr(exc, "body", None)
    message = (body.get("error") or {}).get("message", "") if isinstance(body, dict) else ""
    m = _TPM_LIMIT_REQUESTED_PATTERN.search(message or "")
    if not m:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    return (limit, requested) if requested > limit else None


def _is_hard_payload_limit(exc: OpenAIError) -> bool:
    """True for Groq's flat 'Request Entity Too Large' / {"code":
    "request_too_large", "type": "invalid_request_error"} response -- a hard
    cap on the raw HTTP request body size, structurally different from the
    tokens-per-minute budget _tpm_over_limit() handles (that one is a
    rolling time window; this one has no time dimension at all -- it's just
    "this payload, as bytes, is bigger than the endpoint accepts"). No
    "Limit N, Requested M" numbers are given here to compare, but the
    conclusion is the same: waiting can never help, since the exact same
    oversized request is sent again unchanged. Distinguished from Groq's
    other 413 (TPM/rate_limit_exceeded, handled by _is_rate_or_size_limited)
    by its distinct error code.
    """
    if not isinstance(exc, APIStatusError) or exc.status_code != 413:
        return False
    body = getattr(exc, "body", None)
    return isinstance(body, dict) and body.get("error", {}).get("code") == "request_too_large"


#: Groq's rate_limit_exceeded message often names the actual wait, e.g.
#: "...Limit 12000, Used 9524, Requested 4941. Please try again in
#: 12.325s." -- far more precise than any flat guess, since it reflects
#: exactly how much of the current per-minute window is left.
_RETRY_AFTER_MESSAGE_PATTERN = re.compile(r"try again in\s+([\d.]+)\s*s\b", re.IGNORECASE)


def _rate_limit_wait_seconds(exc: OpenAIError) -> float:
    """Pick how long to wait before retrying `exc`, preferring the most
    precise source available:
      1. A wait time embedded in the error message itself (Groq's
         rate_limit_exceeded messages usually include one) -- a small
         safety margin is added since it's a lower bound, not a guarantee.
      2. A Retry-After response header, if the provider sent one.
      3. RATE_LIMIT_BACKOFF_SECONDS, as a last-resort flat guess.
    """
    body = getattr(exc, "body", None)
    message = (body.get("error") or {}).get("message", "") if isinstance(body, dict) else ""
    m = _RETRY_AFTER_MESSAGE_PATTERN.search(message or "")
    if m:
        try:
            return max(float(m.group(1)) + 1.0, 1.0)
        except ValueError:
            pass

    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    retry_after = header.get("retry-after") if header else None
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except (TypeError, ValueError):
            pass
    return RATE_LIMIT_BACKOFF_SECONDS


def _query_llm_json_array(
    *,
    provider: str,
    model: str | None,
    system_prompt: str,
    prompt: str,
    item_count: int,
    item_label: str,
    validator,
    retries: int,
    retry_delay: float,
    verbose: bool,
    debug: bool = False,
    tpm_budget: int = DEFAULT_TPM_BUDGET,
) -> list[dict]:
    """Shared request/retry/parse/validate loop used by both parameter and
    metric inference -- they only differ in prompts and validation rules.

    Every attempt reserves its estimated token cost against a per-provider
    _RateLimiter shared across the whole process (see this module's "Token
    estimation..." section) BEFORE sending -- so a burst of small, otherwise-
    individually-fine requests can't collectively blow through the account's
    real rolling TPM window the way per-request-only checks would miss.

    `verbose` and `debug` are two separate detail levels, not one: `verbose`
    shows the short per-request header (endpoint/model/prompt size/estimated
    cost) plus the rate-limiter's own wait/no-wait lines -- enough to see
    what's happening without drowning in it. `debug` additionally dumps the
    full request/response text -- implies `verbose` isn't required to also
    be set (debug alone is enough to see everything; passing both is
    harmless and just as verbose as debug alone).
    """
    verbose = verbose or debug
    client = get_client(provider)
    resolved_model = model or PROVIDER_CONFIG[provider]["default_model"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    estimated = estimate_request_tokens(system_prompt, prompt, item_count)
    limiter = _get_rate_limiter(provider, tpm_budget)

    if verbose:
        print(f"\n[{provider}] endpoint       : {PROVIDER_CONFIG[provider]['base_url']}")
        print(f"[{provider}] model          : {resolved_model}")
        print(f"[{provider}] prompt size    : {len(prompt):,} chars, {item_count} {item_label}(s)")
        print(f"[{provider}] estimated cost : ~{estimated:,} tokens (target budget: {tpm_budget:,} TPM)")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        entry = limiter.reserve(estimated, verbose=verbose)
        try:
            if debug:
                print(f"\n--- Request (attempt {attempt}/{retries}) ---")
            start = time.monotonic()
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0,
            )
            elapsed = time.monotonic() - start
            usage = getattr(response, "usage", None)
            limiter.record_actual(entry, usage.total_tokens if usage is not None else None)
            raw_text = response.choices[0].message.content.strip()

            if debug:
                if usage is not None:
                    print(
                        f"--- Response (attempt {attempt}/{retries}, {elapsed:.2f}s, "
                        f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                        f"total={usage.total_tokens} tokens) ---"
                    )
                else:
                    print(f"--- Response (attempt {attempt}/{retries}, {elapsed:.2f}s) ---")
                print(raw_text)
                print("--- end response ---\n")
            elif verbose:
                extra = f", prompt={usage.prompt_tokens} completion={usage.completion_tokens}" if usage else ""
                print(f"[{provider}] response       : {elapsed:.2f}s{extra}")

            text = extract_json_array(raw_text)
            data = json.loads(text)
            return validator(data)

        except json.JSONDecodeError as exc:
            print(f"Attempt {attempt}/{retries}: JSON parse error — {exc}")
            last_error = exc

        except NotFoundError as exc:
            # A 404 here means the model id doesn't exist, or this specific
            # API key/account doesn't have it enabled -- retrying identical
            # requests can't fix that, so fail fast instead of burning all
            # `retries` attempts on the same guaranteed-to-fail call. The
            # request was rejected before generating anything, so release
            # the optimistic reservation instead of holding it against
            # later requests that share this limiter.
            limiter.record_actual(entry, 0)
            raise RuntimeError(
                f"Model '{resolved_model}' was rejected by {provider} (404 model_not_found) -- "
                f"either it doesn't exist, or this API key doesn't have access to it. "
                f"Pass --model to use a different one. To see which models this key CAN access:\n"
                f"  curl -s {PROVIDER_CONFIG[provider]['base_url']}/models "
                f"-H \"Authorization: Bearer ${PROVIDER_CONFIG[provider]['api_key_env']}\" | python3 -m json.tool"
            ) from exc

        except OpenAIError as exc:
            # Same reasoning as NotFoundError above -- every branch below is
            # a rejection that happened before any completion was
            # generated, so nothing was actually consumed.
            limiter.record_actual(entry, 0)
            last_error = exc
            over_limit = _tpm_over_limit(exc)
            if over_limit is not None:
                # This single request alone (item_count items + the full
                # main.cc/problem.hh source embedded in the prompt) needs
                # more tokens than the model's ENTIRE per-minute budget --
                # not "too much recent traffic". No amount of waiting fixes
                # that, so fail immediately instead of burning all
                # `retries` attempts on 65s waits that can't ever succeed.
                limit, requested = over_limit
                raise RuntimeError(
                    f"This request needs ~{requested:,} tokens, but '{resolved_model}' on {provider} is "
                    f"capped at {limit:,} tokens per minute for this account -- a single request over the "
                    f"limit can never succeed no matter how long it waits. Options:\n"
                    f"  - Pass --model to use a model with a higher per-minute budget on your account "
                    f"(check with the models-list command below; on Groq, 'groq/compound'/"
                    f"'groq/compound-mini' typically get a much higher TPM than standard instruct models).\n"
                    f"  - This benchmark's main.cc/problem.hh may just be large -- {item_count} "
                    f"{item_label}(s) were requested in one call; a smaller --scenario-params selection "
                    f"reduces completion size, though the source text itself is the bigger cost here.\n"
                    f"  curl -s {PROVIDER_CONFIG[provider]['base_url']}/models "
                    f"-H \"Authorization: Bearer ${PROVIDER_CONFIG[provider]['api_key_env']}\" | python3 -m json.tool"
                ) from exc
            if _is_hard_payload_limit(exc):
                # A flat request-body-size cap, not a time-windowed budget --
                # unlike a TPM rate limit, there's no "wait it out" here at
                # all: the identical, still-oversized request would just be
                # rejected again immediately. Fail fast, same rationale as
                # the over_limit branch above.
                raise RuntimeError(
                    f"'{resolved_model}' on {provider} rejected this request as too large "
                    f"(413 request_too_large) -- this is a hard cap on the request payload itself, not a "
                    f"per-minute budget, so retrying the identical request can't help. The prompt sent here "
                    f"was ~{len(prompt):,} characters ({item_count} {item_label}(s) plus the full "
                    f"main.cc/problem.hh source). Options:\n"
                    f"  - Pass --model to try a model with a larger request-size limit on your account.\n"
                    f"  - Select fewer parameters/metrics at once (--scenario-params), or trim/split this "
                    f"benchmark's main.cc/problem.hh if it's unusually large -- the embedded source text is "
                    f"almost always the biggest contributor to prompt size here.\n"
                    f"  curl -s {PROVIDER_CONFIG[provider]['base_url']}/models "
                    f"-H \"Authorization: Bearer ${PROVIDER_CONFIG[provider]['api_key_env']}\" | python3 -m json.tool"
                ) from exc
            if _is_rate_or_size_limited(exc):
                wait = _rate_limit_wait_seconds(exc)
                print(
                    f"Attempt {attempt}/{retries} failed — token/rate budget exceeded ({exc}). "
                    f"Waiting {wait:.0f}s for the provider's per-minute budget to reset before retrying "
                    f"(short retries can't fix this -- it needs the window to actually pass)..."
                )
                if attempt < retries:
                    time.sleep(wait)
                continue
            print(f"Attempt {attempt}/{retries} failed — {exc}")

        except ValueError as exc:
            print(f"Attempt {attempt}/{retries} failed — {exc}")
            last_error = exc

        if attempt < retries:
            time.sleep(retry_delay)

    raise RuntimeError(f"All {retries} attempts failed.") from last_error


#: Max candidates sent in a single inference request. Even with the compact
#: per-item cpp_hint/context (see ai.prompts) instead of a full-file dump, a
#: large enough parameter/metric set can still add up -- especially
#: completion size, which scales with item count (each item needs its own
#: semantic_name/datatype/unit/quantityKind/confidence/explanation back).
#: Splitting into batches this size keeps every single request small
#: regardless of how many parameters/metrics a benchmark has. Override via
#: --inference-batch-size.
DEFAULT_BATCH_SIZE = 8


def _batches(items: list, size: int) -> list[list]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


def infer_parameter_metadata(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    *,
    benchmark_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    debug: bool = False,
    known_corrections: list[dict] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tpm_budget: int = DEFAULT_TPM_BUDGET,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata for all
    discovered parameters -- as one request if `candidates` fits within
    `batch_size` AND its estimated token cost fits `tpm_budget`, otherwise
    split into multiple independent requests (each validated against just
    its own slice of `candidates`) and concatenated. Two independent splits
    apply, in order: first a fixed-size chunk (`batch_size` items), then
    each chunk is split further if its own estimated cost alone still
    exceeds `tpm_budget` (see _split_by_token_budget()) -- item count alone
    isn't a reliable proxy for request size, since per-item cpp_hint/context
    excerpts vary. Every request this call makes also reserves its estimated
    cost against a rate limiter shared across the whole process (see
    _RateLimiter), so a run's total request rate stays within `tpm_budget`
    too, not just each individual request's size.

    `known_corrections` (optional -- see ai.corrections.relevant_corrections_for)
    is a list of prior human corrections to similarly named/typed parameters,
    included in the prompt as guidance so the model doesn't repeat a mistake
    a human already fixed once.
    """
    results: list[dict] = []

    def build(batch: list[ParameterCandidate]) -> str:
        return build_prompt(batch, main_cc, problem_hh, benchmark_description, known_corrections)

    for batch in _batches(candidates, batch_size):
        for sub_batch in _split_by_token_budget(batch, build, SYSTEM_PROMPT, tpm_budget, "parameter"):
            prompt = build(sub_batch)
            results.extend(_query_llm_json_array(
                provider=provider,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                prompt=prompt,
                item_count=len(sub_batch),
                item_label="parameter",
                validator=lambda data, sub_batch=sub_batch: validate_metadata(data, sub_batch),
                retries=retries,
                retry_delay=retry_delay,
                verbose=verbose,
                debug=debug,
                tpm_budget=tpm_budget,
            ))
    return results


def infer_metric_metadata(
    candidates: list[MetricCandidate],
    main_cc: str,
    problem_hh: str,
    *,
    benchmark_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    debug: bool = False,
    known_corrections: list[dict] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tpm_budget: int = DEFAULT_TPM_BUDGET,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata -- with SI
    units -- for all discovered output/solution metrics. Batched and
    rate-limited the same way as infer_parameter_metadata() above -- see
    its docstring.

    `known_corrections` -- see infer_parameter_metadata().
    """
    results: list[dict] = []

    def build(batch: list[MetricCandidate]) -> str:
        return build_metric_prompt(batch, main_cc, problem_hh, benchmark_description, known_corrections)

    for batch in _batches(candidates, batch_size):
        for sub_batch in _split_by_token_budget(batch, build, METRIC_SYSTEM_PROMPT, tpm_budget, "metric"):
            prompt = build(sub_batch)
            results.extend(_query_llm_json_array(
                provider=provider,
                model=model,
                system_prompt=METRIC_SYSTEM_PROMPT,
                prompt=prompt,
                item_count=len(sub_batch),
                item_label="metric",
                validator=lambda data, sub_batch=sub_batch: validate_metric_metadata(data, sub_batch),
                retries=retries,
                retry_delay=retry_delay,
                verbose=verbose,
                debug=debug,
                tpm_budget=tpm_budget,
            ))
    return results


