# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.inference

LLM provider configuration (Groq/OpenAI) and the request/retry/validate
loop that turns discovered parameter/metric candidates into inferred
semantic metadata.
"""

from __future__ import annotations

import json
import os
import re
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


def _rate_limit_wait_seconds(exc: OpenAIError) -> float:
    """Honor a Retry-After response header if the provider sent one,
    otherwise fall back to RATE_LIMIT_BACKOFF_SECONDS.
    """
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
) -> list[dict]:
    """Shared request/retry/parse/validate loop used by both parameter and
    metric inference -- they only differ in prompts and validation rules.
    """
    client = get_client(provider)
    resolved_model = model or PROVIDER_CONFIG[provider]["default_model"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if verbose:
        print(f"\n[{provider}] endpoint   : {PROVIDER_CONFIG[provider]['base_url']}")
        print(f"[{provider}] model      : {resolved_model}")
        print(f"[{provider}] prompt size: {len(prompt):,} chars, {item_count} {item_label}(s)")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(f"\n--- Request (attempt {attempt}/{retries}) ---")
            start = time.monotonic()
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0,
            )
            elapsed = time.monotonic() - start
            raw_text = response.choices[0].message.content.strip()

            if verbose:
                usage = getattr(response, "usage", None)
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
            # `retries` attempts on the same guaranteed-to-fail call.
            raise RuntimeError(
                f"Model '{resolved_model}' was rejected by {provider} (404 model_not_found) -- "
                f"either it doesn't exist, or this API key doesn't have access to it. "
                f"Pass --model to use a different one. To see which models this key CAN access:\n"
                f"  curl -s {PROVIDER_CONFIG[provider]['base_url']}/models "
                f"-H \"Authorization: Bearer ${PROVIDER_CONFIG[provider]['api_key_env']}\" | python3 -m json.tool"
            ) from exc

        except OpenAIError as exc:
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
    known_corrections: list[dict] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata for all
    discovered parameters -- as one request if `candidates` fits within
    `batch_size`, otherwise split into multiple independent requests (each
    validated against just its own slice of `candidates`) and concatenated.
    Splitting keeps every single request's size small and predictable
    regardless of how many parameters a benchmark has.

    `known_corrections` (optional -- see ai.corrections.relevant_corrections_for)
    is a list of prior human corrections to similarly named/typed parameters,
    included in the prompt as guidance so the model doesn't repeat a mistake
    a human already fixed once.
    """
    results: list[dict] = []
    for batch in _batches(candidates, batch_size):
        prompt = build_prompt(batch, main_cc, problem_hh, benchmark_description, known_corrections)
        results.extend(_query_llm_json_array(
            provider=provider,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            item_count=len(batch),
            item_label="parameter",
            validator=lambda data, batch=batch: validate_metadata(data, batch),
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
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
    known_corrections: list[dict] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata -- with SI
    units -- for all discovered output/solution metrics. Batched the same
    way as infer_parameter_metadata() above -- see its docstring.

    `known_corrections` -- see infer_parameter_metadata().
    """
    results: list[dict] = []
    for batch in _batches(candidates, batch_size):
        prompt = build_metric_prompt(batch, main_cc, problem_hh, benchmark_description, known_corrections)
        results.extend(_query_llm_json_array(
            provider=provider,
            model=model,
            system_prompt=METRIC_SYSTEM_PROMPT,
            prompt=prompt,
            item_count=len(batch),
            item_label="metric",
            validator=lambda data, batch=batch: validate_metric_metadata(data, batch),
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
        ))
    return results


