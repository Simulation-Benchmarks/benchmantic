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

from openai import APIStatusError, BadRequestError, NotFoundError, OpenAI, OpenAIError, RateLimitError

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


# =============================================================================
# Automatic model fallback on tokens-per-day (TPD) exhaustion
# =============================================================================
#
# Groq (and, in principle, any OpenAI-compatible provider) tracks its daily
# token quota PER MODEL, not as one account-wide pool shared across every
# model on the key -- confirmed by Groq's own per-model rate-limit table
# (RPM/RPD/TPM/TPD/ASH/ASD columns, one row per model) and independently by
# this same file's PROVIDER_CONFIG comment above, which already documents
# wildly different per-model TPM caps on the same account/tier (8K vs 70K).
# That means a model hitting its TPD cap is a property of THAT model, not
# the account -- so switching to a different model this key can access is a
# genuinely different, still-full quota, not a no-op. This section makes
# that switch automatic instead of requiring the user to notice the error,
# pick a new --model by hand, and re-run.
#
# _EXHAUSTED_MODELS / _ACTIVE_MODEL are process-lifetime, per-provider state
# (same pattern as _RATE_LIMITERS above) so a switch made for one batch's
# parameter-inference call carries over to every later batch and to
# metric-inference too, within the same run -- not just the one request
# that triggered it.

#: Per provider, the set of model ids this run has already exhausted the
#: daily quota for -- never retried again automatically within this process.
_EXHAUSTED_MODELS: dict[str, set[str]] = {}

#: Per provider, the model id currently in use after an automatic fallback
#: switch (overrides PROVIDER_CONFIG's default_model, but NOT an explicit
#: --model the user passed -- see _query_llm_json_array).
_ACTIVE_MODEL: dict[str, str] = {}

#: Substrings that mark a model id as not a general chat-completion model
#: (transcription, text-to-speech, moderation/guard, embeddings) -- these
#: exist in a Groq/OpenAI /v1/models listing alongside chat models but can't
#: serve this pipeline's chat.completions.create() calls, so automatic
#: fallback discovery skips them. NOTE: this list is necessarily incomplete
#: -- Groq's catalog includes TTS/voice models with no consistent naming
#: signal at all (e.g. "canopylabs/orpheus-arabic-saudi", which contains
#: none of these substrings and was picked automatically before this
#: comment existed, then failed a real request with a 400
#: "model_terms_required" every attempt). Catching that class of failure
#: at the exception level (see _is_model_unusable() below) rather than
#: trying to enumerate every non-chat model by name upfront is the more
#: robust fix -- this hint list is a cheap first pass, not the real
#: safety net.
_NON_CHAT_MODEL_HINTS = ("whisper", "tts", "guard", "moderation", "embed", "safety", "orpheus", "canopylabs")

#: Substrings marking a model as a small and/or narrowly-specialized
#: chat model -- still usable in principle (chat.completions.create()
#: will accept it), but a poor fit for this pipeline's actual task: pulling
#: structured JSON metadata out of C++ source given a fairly long prompt
#: (main.cc/problem.hh excerpts, prior corrections, several items per
#: batch). In practice these models are far more likely to produce
#: malformed JSON, drop required fields, or otherwise fail validation --
#: observed directly: Groq's "allam-2-7b" (a 7B, primarily Arabic-focused
#: instruct model) was auto-selected here purely because "allam" sorts
#: early alphabetically, and then failed identically on all 3 attempts
#: with the exact same JSON syntax error (deterministic at temperature=0,
#: so retrying changed nothing). These ids are DEPRIORITIZED, not
#: excluded outright -- if nothing else is available, one of these is
#: still tried rather than giving up, since a low-capability model that
#: might work beats stopping the run.
_SMALL_OR_SPECIALIZED_MODEL_HINTS = (
    "allam", "distil", "1b", "2b", "3b", "4b", "7b", "8b", "9b",
)

#: Known-good general-purpose Groq chat/instruct models, in preference
#: order. Tried in this order -- first one present in the account's own
#: client.models.list() result (and not already excluded) wins -- before
#: falling back to the general/narrow heuristic ranking below for
#: anything not on this list.
#:
#: Why a curated list on top of live discovery, not instead of it: two
#: real failures on the same account showed pure heuristic ranking over
#: whatever models.list() happens to return isn't reliable enough on
#: Groq's full catalog, which mixes general chat models in with TTS,
#: region/language-specialized, and terms-gated ones that carry no
#: consistent naming signal to filter on ("canopylabs/orpheus-arabic-
#: saudi" has none of _NON_CHAT_MODEL_HINTS's substrings and still isn't
#: usable here). A short list of models known to actually work well for
#: this pipeline's structured-JSON task is a much stronger signal than
#: "sorts alphabetically early" or "isn't obviously small". This list is
#: intentionally NOT exhaustive -- it doesn't need to be, since anything
#: not on it still gets tried as a fallback, just ranked by the general/
#: narrow heuristic instead of a known-good preference.
_PREFERRED_MODEL_ORDER = [
    "llama-3.3-70b-versatile",
    "groq/compound-mini",
    "groq/compound",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]


def _discover_alternate_model(client: OpenAI, provider: str, exclude: set[str]) -> str | None:
    """Ask the provider what models this API key can actually access
    (GET /v1/models, exposed by the openai-python SDK as client.models.list()
    -- works against Groq's OpenAI-compatible endpoint too) and return one
    not in `exclude` and not an obviously non-chat model, or None if none is
    available (models.list() itself failed, or nothing usable is left).

    Selection order:
      1. The first _PREFERRED_MODEL_ORDER entry present among the
         remaining candidates -- a short, curated list of models already
         known to work well for this pipeline's task (see that list's
         comment for why discovery alone isn't reliable enough here).
      2. Otherwise, among whatever's left: general-purpose models before
         small/narrowly-specialized ones (_SMALL_OR_SPECIALIZED_MODEL_HINTS),
         sorted alphabetically within each group for a deterministic,
         reproducible choice.

    Still queries live rather than ONLY using a hardcoded list -- this
    file's own comments already note Groq model availability varies per
    account/key, so an account with none of the preferred models (or a
    non-Groq provider) still gets a reasonable pick instead of nothing.
    """
    try:
        listing = client.models.list()
    except OpenAIError:
        return None
    candidates = []
    for entry in getattr(listing, "data", None) or []:
        model_id = getattr(entry, "id", None)
        if not model_id or model_id in exclude:
            continue
        if any(hint in model_id.lower() for hint in _NON_CHAT_MODEL_HINTS):
            continue
        candidates.append(model_id)
    if not candidates:
        return None

    candidate_set = set(candidates)
    for preferred in _PREFERRED_MODEL_ORDER:
        if preferred in candidate_set:
            return preferred

    def is_small_or_specialized(model_id: str) -> bool:
        return any(hint in model_id.lower() for hint in _SMALL_OR_SPECIALIZED_MODEL_HINTS)

    general = sorted(m for m in candidates if not is_small_or_specialized(m))
    narrow = sorted(m for m in candidates if is_small_or_specialized(m))
    return (general + narrow)[0]


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



def _error_body(exc: OpenAIError) -> dict:
    """Returns the {"message": ..., "type": ..., "code": ...} error dict
    for `exc`, normalizing across the two shapes openai-python's client
    code can end up handing an exception's `.body`.

    The server always sends the FULL {"error": {...}} envelope, but
    OpenAI._make_status_error() (this SDK's status-error constructor)
    unwraps that itself before building the exception -- `data =
    body.get("error", body) if is_mapping(body) else body` -- so by the
    time code here ever sees `exc.body`, it's normally already the INNER
    dict, with no "error" key one level up. Every helper below used to
    assume the un-unwrapped shape (`body["error"]["message"]`), which
    silently returned nothing for a real response instead of raising --
    `.get("error")` on a dict with no "error" key just returns None, no
    exception -- so this was never caught by normal error handling, only
    by the symptom of daily-quota/oversized-request detection never
    firing against a live API and every 429 falling through to the
    generic short-backoff retry path instead, no matter what it actually
    was. Handling BOTH shapes here (checking for a nested "error" dict
    first, falling back to the body itself) is deliberately defensive
    against this exact class of mistake happening again if a future SDK
    version changes which shape it hands over.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    inner = body.get("error")
    return inner if isinstance(inner, dict) else body


#: Error codes meaning "this specific model can't serve requests on this
#: account right now, for a reason no amount of retrying (or even waiting)
#: fixes" -- as opposed to a rate/token budget, which resolves with time.
#: "model_terms_required" is the one actually observed (Groq's org admin
#: hasn't accepted that model's terms yet -- see console.groq.com/
#: playground?model=... in the error message); the other two are
#: plausible siblings for a model pulled from an account's catalog that
#: isn't actually invocable (retired, or temporarily disabled) -- harmless
#: to check for even if never seen in practice, since an unmatched code
#: just falls through to the generic retry path unchanged.
_MODEL_UNUSABLE_CODES = {"model_terms_required", "model_not_active", "model_decommissioned"}


def _is_model_unusable(exc: OpenAIError) -> bool:
    return _error_body(exc).get("code") in _MODEL_UNUSABLE_CODES


def _is_rate_or_size_limited(exc: OpenAIError) -> bool:
    """True for a 429 (RateLimitError) or Groq's 413 'request too large for
    tokens-per-minute budget' response -- both mean "the provider rejected
    this on token/rate budget grounds", not "something is wrong with the
    request itself". Groq's TPM-exceeded case comes back as a generic
    APIStatusError (413 has no dedicated openai-python exception class), so
    it's detected via the response body's {"code": "rate_limit_exceeded"}
    instead of the exception type alone (see _error_body() for why this
    reads `exc.body` directly rather than assuming a nested "error" key).
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        if _error_body(exc).get("code") == "rate_limit_exceeded":
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
    message = _error_body(exc).get("message", "")
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
    return _error_body(exc).get("code") == "request_too_large"


#: Groq's rate_limit_exceeded message names the actual wait, but NOT always
#: as a bare "12.325s" -- once the wait is more than a minute (routine for a
#: daily-quota message; see _DAILY_LIMIT_PATTERN below) it switches to a
#: compound "33m47.808s" or "1h12m45.79s" form instead, with no space
#: between the h/m/s components. Each component is optional on its own
#: (a sub-minute wait is just "12.325s", no "m"/"h" at all), so every piece
#: is captured separately and combined below rather than assuming seconds
#: is always the only or the last-in-message number.
_RETRY_AFTER_MESSAGE_PATTERN = re.compile(
    r"try again in\s+(?:([\d.]+)\s*h)?\s*(?:([\d.]+)\s*m(?!s))?\s*(?:([\d.]+)\s*s)?", re.IGNORECASE,
)

#: Groq's daily-quota message reads "...on tokens per day (TPD): Limit
#: 100000, Used 99619, Requested 2728. Please try again in 33m47.808s.".
#: This is a fundamentally different kind of limit from the per-minute (TPM)
#: budget the rest of this module is built around: it doesn't reset on its
#: own within a single interactive run (the wait is routinely 10s of
#: minutes to multiple hours, not the ~60s a TPM window needs), so treating
#: it like any other rate_limit_exceeded and retrying with a short backoff
#: -- as this code used to, before this pattern was recognized at all --
#: just burns through `retries` attempts that are all guaranteed to fail
#: identically, while also misreporting the wait as "the provider's
#: per-minute budget" resetting soon when it's actually a day-scale quota.
_DAILY_LIMIT_PATTERN = re.compile(r"tokens per day\s*\(TPD\)", re.IGNORECASE)

#: Groq's rate-limit messages open with "Rate limit reached for model
#: `<id>`" -- that `<id>` is the model that ACTUALLY ran out of TPD quota,
#: which is not always the model this code requested. In particular,
#: Groq's "compound"/"compound-mini" systems (see PROVIDER_CONFIG's
#: comment on them) are agentic wrappers that internally delegate to an
#: underlying instruct model (observed in practice: requesting
#: "groq/compound-mini" produced a TPD error naming
#: "llama-3.3-70b-versatile", the model compound-mini was routing to, not
#: compound-mini itself) -- so the quota that's actually exhausted belongs
#: to that underlying model, not the wrapper id in `resolved_model`.
#: Extracting it lets automatic fallback exclude BOTH ids from the next
#: pick, rather than switching to a different wrapper that may delegate to
#: the very same already-exhausted underlying model.
_MESSAGE_MODEL_PATTERN = re.compile(r"for model `([^`]+)`")


def _message_named_model(message: str) -> str | None:
    m = _MESSAGE_MODEL_PATTERN.search(message or "")
    return m.group(1) if m else None


def _parse_message_wait_seconds(message: str) -> float | None:
    """Parse Groq's "please try again in ..." wait, in either its plain
    "12.325s" form or its compound "33m47.808s" / "1h12m45.79s" form --
    returns None if the message has no such phrase at all (a small safety
    margin is added on top of whatever's parsed, since it's a lower bound
    on the provider's own countdown, not a guarantee).
    """
    m = _RETRY_AFTER_MESSAGE_PATTERN.search(message or "")
    if not m or not any(m.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in m.groups())
    return hours * 3600 + minutes * 60 + seconds + 1.0


def _format_wait(seconds: float) -> str:
    """"4059.8" -> "1h 7m 40s" -- for a human-readable wait in an error
    message; drops leading zero components (a 47s wait reads as "47s", not
    "0h 0m 47s")."""
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _is_daily_limit(exc: OpenAIError) -> float | None:
    """If `exc` is Groq's tokens-per-day (TPD) quota being exhausted,
    returns how many seconds the message says to wait (falling back to a
    generic non-zero placeholder if that couldn't be parsed) -- otherwise
    None. See _DAILY_LIMIT_PATTERN above for why this needs its own
    fail-fast path rather than going through the normal short-backoff
    retry that per-minute (TPM) limits use.
    """
    message = _error_body(exc).get("message", "")
    if not _DAILY_LIMIT_PATTERN.search(message or ""):
        return None
    return _parse_message_wait_seconds(message) or RATE_LIMIT_BACKOFF_SECONDS


def _rate_limit_wait_seconds(exc: OpenAIError) -> float:
    """Pick how long to wait before retrying `exc`, preferring the most
    precise source available:
      1. A wait time embedded in the error message itself (Groq's
         rate_limit_exceeded messages usually include one, in either the
         plain or compound h/m/s form -- see _parse_message_wait_seconds).
      2. A Retry-After response header, if the provider sent one.
      3. RATE_LIMIT_BACKOFF_SECONDS, as a last-resort flat guess.

    Only meant for waits actually worth sleeping through inline (a
    per-minute budget resetting) -- a daily quota's much longer wait is
    caught separately by _is_daily_limit() and fails fast instead of
    reaching this function's caller at all.
    """
    message = _error_body(exc).get("message", "")
    parsed = _parse_message_wait_seconds(message)
    if parsed is not None:
        return parsed

    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    retry_after = header.get("retry-after") if header else None
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except (TypeError, ValueError):
            pass
    return RATE_LIMIT_BACKOFF_SECONDS


class _DailyQuotaExhausted(Exception):
    """Internal signal (never escapes this module) raised by
    _query_llm_json_array_one_model() when a model hits its tokens-per-day
    quota, so the outer _query_llm_json_array() can decide whether to
    switch models and retry, versus giving up -- see that function.
    """

    def __init__(self, resolved_model: str, wait_seconds: float, original: OpenAIError, message_model: str | None = None):
        super().__init__(str(original))
        self.resolved_model = resolved_model
        self.original = original
        #: The model id Groq's own error message names as exhausted, if it
        #: differs from `resolved_model` (the id this code requested) --
        #: see _MESSAGE_MODEL_PATTERN above for why these two can diverge.
        self.message_model = message_model
        self.wait_seconds = wait_seconds


class _ModelOutputInvalid(Exception):
    """Internal signal (never escapes this module) raised by
    _query_llm_json_array_one_model() when a model consistently fails to
    produce a response this pipeline can use -- every one of `retries`
    attempts either wasn't valid JSON (json.JSONDecodeError) or didn't
    pass schema validation (ValueError from validator(), e.g. missing
    required fields) -- as opposed to a provider-side rejection
    (OpenAIError), which raises RuntimeError directly instead (that's an
    infrastructure problem a different model can't be assumed to fix).

    This is a genuinely different failure mode from _DailyQuotaExhausted,
    but worth the same treatment: at temperature=0 a model that fails
    this way on attempt 1 will fail identically on attempts 2 and 3 (all
    3 retries just re-send the exact same request), so the retries
    accomplish nothing on their own -- observed directly when an
    automatically-selected fallback model ("allam-2-7b", small and not
    well-suited to structured JSON extraction from source code) hit the
    exact same JSON syntax error on all 3 attempts. Trying a different
    model is a real chance at success, not just noise, so
    _query_llm_json_array() gets the same opportunity to switch here as
    it does for a daily-quota exhaustion.
    """

    def __init__(self, resolved_model: str, original: Exception):
        super().__init__(str(original))
        self.resolved_model = resolved_model
        self.original = original


class _ModelUnusable(Exception):
    """Internal signal (never escapes this module) raised by
    _query_llm_json_array_one_model() the moment a model turns out unable
    to handle THIS request at all, for a structural reason no amount of
    retrying (or even waiting) fixes -- covering four distinct rejections
    observed in practice, all given the same treatment because they share
    the same fix (try a different model) and the same non-fix (retrying
    the identical request against the identical model):
      - 404: the model doesn't exist, or isn't enabled for this key.
      - 400 with a code in _MODEL_UNUSABLE_CODES (e.g.
        "model_terms_required": Groq's org admin hasn't accepted that
        specific model's terms yet -- observed with
        "canopylabs/orpheus-arabic-saudi", a TTS model discovery picked up
        with no way to have known from its id alone that it needed this).
      - A single request's estimated cost exceeds a model's ENTIRE
        per-minute (TPM) budget -- not "too much recent traffic", the
        model's whole budget is smaller than this one request.
      - 413 request_too_large: a flat cap on the request's raw byte size,
        smaller than what this prompt needs -- observed directly:
        automatically switching from 'groq/compound-mini' to its sibling
        'groq/compound' (for an unrelated TPD reason) landed on a model
        whose payload cap was too small for the exact same prompt that
        compound-mini had accepted fine.
    Raised on the FIRST attempt, not after exhausting `retries` -- unlike
    a token/rate-limit response, nothing about any of these four changes
    on a second identical request, so retrying 2 more times would just
    repeat the exact same rejection for no benefit (this was actually
    happening before this class existed: 3 identical "requires terms
    acceptance" errors in a row before finally giving up).
    """

    def __init__(self, resolved_model: str, original: OpenAIError):
        super().__init__(str(original))
        self.resolved_model = resolved_model
        self.original = original


def _query_llm_json_array_one_model(
    *,
    provider: str,
    resolved_model: str,
    client: OpenAI,
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
    """Request/retry/parse/validate loop for ONE fixed model. Raises
    _DailyQuotaExhausted (instead of RuntimeError) if `resolved_model` hits
    its tokens-per-day quota, so the caller (_query_llm_json_array) can try
    switching to a different model instead of failing outright -- every
    other failure mode here still raises RuntimeError/re-raises directly,
    since only a TPD quota is a property of the MODEL that another model on
    the same account/key can route around (see the module-level "Automatic
    model fallback" comment above).

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
            # later requests that share this limiter. This used to raise
            # RuntimeError directly; now it's the same _ModelUnusable
            # signal the "model_terms_required"-style 400s below use, so
            # the caller gets the same chance to switch models
            # automatically instead of the whole run just stopping here.
            limiter.record_actual(entry, 0)
            raise _ModelUnusable(resolved_model, exc) from exc

        except BadRequestError as exc:
            if _is_model_unusable(exc):
                # e.g. Groq's "model_terms_required" -- this model needs an
                # org admin to accept its terms before it can be used at
                # all, which no amount of retrying (or even switching
                # --inference-tpm-budget/batch-size) changes. Same
                # fail-fast-and-let-the-caller-try-another-model treatment
                # as NotFoundError above, for the same reason.
                limiter.record_actual(entry, 0)
                raise _ModelUnusable(resolved_model, exc) from exc
            # Any other 400 (a genuine bad request -- malformed messages,
            # an unsupported parameter, etc.) falls through to the
            # generic OpenAIError handling below, same as before this
            # class existed.
            limiter.record_actual(entry, 0)
            last_error = exc
            print(f"Attempt {attempt}/{retries} failed — {exc}")

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
                # This model's per-minute budget can never fit this
                # request no matter how long it waits -- but a DIFFERENT
                # model on the same account may have a much higher TPM
                # cap (this file's own PROVIDER_CONFIG comment notes
                # exactly that spread on Groq), so this is the same
                # "this model can't handle this request, try another"
                # situation as _ModelUnusable's other cases, not a dead
                # end -- let the caller attempt a switch instead of
                # failing immediately.
                raise _ModelUnusable(resolved_model, exc) from exc
            daily_wait = _is_daily_limit(exc)
            if daily_wait is not None:
                # A tokens-per-day quota, not the per-minute budget the
                # rest of this loop's retries assume -- see
                # _DAILY_LIMIT_PATTERN above. The wait is routinely 10s of
                # minutes to multiple hours, nothing a retry loop inside
                # one run should sit through blindly. Unlike a per-minute
                # limit, this one IS worth trying a different model for --
                # Groq tracks TPD per model, not account-wide (see the
                # "Automatic model fallback" section above) -- so raise a
                # distinct signal instead of RuntimeError directly and let
                # _query_llm_json_array() decide whether switching models
                # can route around it before giving up.
                message = _error_body(exc).get("message", "")
                message_model = _message_named_model(message)
                raise _DailyQuotaExhausted(
                    resolved_model, daily_wait, exc,
                    message_model=message_model if message_model != resolved_model else None,
                ) from exc
            if _is_hard_payload_limit(exc):
                # A flat request-body-size cap, not a time-windowed budget --
                # unlike a TPM rate limit, there's no "wait it out" here at
                # all: the identical, still-oversized request would just be
                # rejected again immediately by THIS model. Observed
                # directly: switching from 'groq/compound-mini' to
                # 'groq/compound' (an automatic fallback for a DIFFERENT
                # reason, its sibling wrapper) landed on a model with a
                # smaller request-size cap that then rejected the exact
                # same prompt outright. A different model may well have a
                # larger cap, so -- same as the over_limit case above --
                # this is "this model can't handle this request", not
                # "nothing can"; let the caller try switching first.
                raise _ModelUnusable(resolved_model, exc) from exc
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

    if isinstance(last_error, (json.JSONDecodeError, ValueError)):
        # Every attempt failed on the MODEL's OUTPUT (malformed JSON, or
        # JSON that didn't pass schema validation) rather than on a
        # provider-side rejection -- a different model gets a genuine shot
        # at this prompt, so let the caller try switching instead of
        # failing immediately. See _ModelOutputInvalid's docstring.
        raise _ModelOutputInvalid(resolved_model, last_error) from last_error
    raise RuntimeError(f"All {retries} attempts failed.") from last_error


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
    allow_model_fallback: bool = True,
) -> list[dict]:
    """Wraps _query_llm_json_array_one_model() with automatic model
    fallback: if the model in use hits its tokens-per-day quota, and
    `allow_model_fallback` is set, try discovering a different model this
    API key can access (see _discover_alternate_model()) and retry with it
    -- carrying that switch forward (via _ACTIVE_MODEL) to every later call
    this run makes to the same provider, not just this one -- instead of
    immediately failing the whole run. Falls back to today's behavior
    (fail fast with a clear message) once every reachable model has been
    tried, or if fallback is disabled, or if discovery itself doesn't turn
    up an untried model.

    `model` -- an explicit --model the caller pinned -- is always tried
    first and is never itself replaced by _ACTIVE_MODEL's last automatic
    pick from a PRIOR call; but if that pinned model is the one that hits
    its TPD quota, it's just as eligible for automatic fallback as the
    default would be, since the alternative is failing the run outright
    for exactly the problem this feature exists to solve.
    """
    client = get_client(provider)
    tried: set[str] = set()
    last_daily_error: _DailyQuotaExhausted | None = None
    last_failure_kind: str | None = None  # "daily" or "output" -- whichever happened LAST, for the final message

    if verbose or debug:
        # Deliberately printed every call (not just once per process) so
        # it's impossible to miss in a log, and so it's an unambiguous way
        # to tell whether a given ai/inference.py copy actually has this
        # feature at all -- if this line is absent from your output, the
        # file being imported isn't this one (stale copy, or an old
        # __pycache__/*.pyc being reused instead of the current source).
        print(
            f"[{provider}] model fallback : "
            f"{'enabled -- will try another model on a tokens-per-day quota' if allow_model_fallback else 'disabled (--no-model-fallback)'}"
        )

    last_output_error: _ModelOutputInvalid | None = None
    last_unusable_error: _ModelUnusable | None = None

    while True:
        resolved_model = model or _ACTIVE_MODEL.get(provider) or PROVIDER_CONFIG[provider]["default_model"]
        tried.add(resolved_model)
        try:
            return _query_llm_json_array_one_model(
                provider=provider,
                resolved_model=resolved_model,
                client=client,
                system_prompt=system_prompt,
                prompt=prompt,
                item_count=item_count,
                item_label=item_label,
                validator=validator,
                retries=retries,
                retry_delay=retry_delay,
                verbose=verbose,
                debug=debug,
                tpm_budget=tpm_budget,
            )
        except _DailyQuotaExhausted as exc:
            last_daily_error = exc
            last_failure_kind = "daily"
            exhausted_ids = {exc.resolved_model}
            if exc.message_model:
                # The id actually named in Groq's error can differ from
                # what we requested -- e.g. requesting "groq/compound-mini"
                # (an agentic wrapper) surfaced a TPD error naming
                # "llama-3.3-70b-versatile", the underlying model it
                # delegated to. Exclude both, or discovery could "switch"
                # to a different wrapper that delegates to the very same
                # already-exhausted underlying model and fail identically.
                exhausted_ids.add(exc.message_model)
            _EXHAUSTED_MODELS.setdefault(provider, set()).update(exhausted_ids)
            tried.update(exhausted_ids)
            if not allow_model_fallback:
                break
            exclude = tried | _EXHAUSTED_MODELS.get(provider, set())
            alternate = _discover_alternate_model(client, provider, exclude)
            if alternate is None:
                break
            named_note = f" (which delegates to '{exc.message_model}')" if exc.message_model else ""
            print(
                f"'{exc.resolved_model}'{named_note} on {provider} has hit its tokens-per-day quota (retry "
                f"in {_format_wait(exc.wait_seconds)}) -- switching to '{alternate}' for the rest of this "
                f"run instead of waiting or stopping.",
                file=sys.stderr,
            )
            _ACTIVE_MODEL[provider] = alternate
            # Loop again with the new model, fresh `retries` attempts of
            # its own -- one model's exhausted quota shouldn't cost the
            # next model any of its own retry budget.
        except _ModelOutputInvalid as exc:
            last_output_error = exc
            last_failure_kind = "output"
            _EXHAUSTED_MODELS.setdefault(provider, set()).add(exc.resolved_model)
            tried.add(exc.resolved_model)
            if not allow_model_fallback:
                break
            exclude = tried | _EXHAUSTED_MODELS.get(provider, set())
            alternate = _discover_alternate_model(client, provider, exclude)
            if alternate is None:
                break
            print(
                f"'{exc.resolved_model}' on {provider} produced unusable output on all {retries} attempts "
                f"({exc.original}) -- switching to '{alternate}' for the rest of this run instead of "
                f"giving up (retrying the SAME model wouldn't help: it's deterministic at temperature=0, so "
                f"identical requests just fail identically).",
                file=sys.stderr,
            )
            _ACTIVE_MODEL[provider] = alternate
        except _ModelUnusable as exc:
            last_unusable_error = exc
            last_failure_kind = "unusable"
            _EXHAUSTED_MODELS.setdefault(provider, set()).add(exc.resolved_model)
            tried.add(exc.resolved_model)
            if not allow_model_fallback:
                break
            exclude = tried | _EXHAUSTED_MODELS.get(provider, set())
            alternate = _discover_alternate_model(client, provider, exclude)
            if alternate is None:
                break
            print(
                f"'{exc.resolved_model}' on {provider} can't handle this request ({exc.original}) -- "
                f"switching to '{alternate}' for the rest of this run instead of giving up.",
                file=sys.stderr,
            )
            _ACTIVE_MODEL[provider] = alternate

    # Every reachable model either isn't accessible, is already known
    # unusable, or fallback itself is disabled -- fail the same way this
    # always has, but naming every model this run actually tried so the
    # message reflects what was actually attempted, not just the last one.
    # Priority when multiple kinds of failure occurred across different
    # models in the same call (rare, but possible if the run cycles
    # through several unsuitable/exhausted
    # models) -- an "unusable output" failure is more actionable to report
    # than "hit its quota", since the corresponding option list differs.
    tried_list = ", ".join(f"'{m}'" for m in sorted(tried))
    if last_failure_kind == "output":
        raise RuntimeError(
            f"{tried_list} on {provider} all produced unusable output (invalid JSON, or JSON that failed "
            f"schema validation) for this request -- retrying the same model again wouldn't help (it's "
            f"deterministic at temperature=0). Options:\n"
            f"  - Pass --model to name a specific model known to handle this reliably on your account "
            f"(larger general-purpose instruct models tend to do better on structured JSON extraction than "
            f"small or narrowly-specialized ones).\n"
            f"  - Pass --fallback-on-error to fill any still-uninferred items with a low-confidence "
            f"placeholder instead of stopping the run, and fix them by hand in review.\n"
            f"  Last error (from '{last_output_error.resolved_model}'): {last_output_error.original}"
        ) from last_output_error.original
    if last_failure_kind == "unusable":
        raise RuntimeError(
            f"{tried_list} on {provider} couldn't handle this request (model doesn't exist / isn't "
            f"enabled for this key, needs an org admin to accept its terms first, or this request's size "
            f"exceeds that model's per-minute-token or per-request-payload cap -- see the error below for "
            f"which). Options:\n"
            f"  - Pass --model to name a model you know this account can actually use for a request this size.\n"
            f"  - This benchmark's main.cc/problem.hh may just be large -- a smaller --scenario-params "
            f"selection reduces prompt size, though the embedded source text is usually the bigger cost.\n"
            f"  - Check which models this key can access:\n"
            f"    curl -s {PROVIDER_CONFIG[provider]['base_url']}/models "
            f"-H \"Authorization: Bearer ${PROVIDER_CONFIG[provider]['api_key_env']}\" | python3 -m json.tool\n"
            f"  - Pass --fallback-on-error to fill any still-uninferred items with a low-confidence "
            f"placeholder instead of stopping the run, and fix them by hand in review.\n"
            f"  Last error (from '{last_unusable_error.resolved_model}'): {last_unusable_error.original}"
        ) from last_unusable_error.original
    raise RuntimeError(
        f"{tried_list} on {provider} {'all hit their' if len(tried) > 1 else 'has hit its'} "
        f"tokens-per-day quota -- not a per-minute rate limit, so short retries can't fix it; the provider "
        f"says to try '{last_daily_error.resolved_model}' again in {_format_wait(last_daily_error.wait_seconds)}. "
        f"Options:\n"
        f"  - Wait for the daily quota to reset (see the exact time in the error below), then re-run the "
        f"same command -- already-cached parameters/metrics won't be re-queried.\n"
        f"  - Pass --provider openai (with OPENAI_API_KEY set) to use a different account's quota for the "
        f"rest of this run.\n"
        f"  - Pass --fallback-on-error to fill any still-uninferred items with a low-confidence placeholder "
        f"instead of stopping the run, and fix them by hand in review.\n"
        f"  Original error: {last_daily_error.original}"
    ) from last_daily_error.original


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
    allow_model_fallback: bool = True,
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

    `allow_model_fallback` -- see _query_llm_json_array(). Left on by
    default so a model hitting its daily quota mid-run doesn't fail
    parameter inference outright; pass False to force today's fail-fast
    behavior instead (e.g. --no-model-fallback).
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
                allow_model_fallback=allow_model_fallback,
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
    allow_model_fallback: bool = True,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata -- with SI
    units -- for all discovered output/solution metrics. Batched and
    rate-limited the same way as infer_parameter_metadata() above -- see
    its docstring.

    `known_corrections` -- see infer_parameter_metadata().
    `allow_model_fallback` -- see infer_parameter_metadata() / _query_llm_json_array().
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
                allow_model_fallback=allow_model_fallback,
            ))
    return results


