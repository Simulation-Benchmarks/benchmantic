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
import time
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

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
        "default_model": "llama-3.3-70b-versatile",
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

        except (OpenAIError, ValueError) as exc:
            print(f"Attempt {attempt}/{retries} failed — {exc}")
            last_error = exc

        if attempt < retries:
            time.sleep(retry_delay)

    raise RuntimeError(f"All {retries} attempts failed.") from last_error


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
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata for all discovered parameters."""
    prompt = build_prompt(candidates, main_cc, problem_hh, benchmark_description)
    return _query_llm_json_array(
        provider=provider,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        item_count=len(candidates),
        item_label="parameter",
        validator=lambda data: validate_metadata(data, candidates),
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )


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
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata -- with SI
    units -- for all discovered output/solution metrics.
    """
    prompt = build_metric_prompt(candidates, main_cc, problem_hh, benchmark_description)
    return _query_llm_json_array(
        provider=provider,
        model=model,
        system_prompt=METRIC_SYSTEM_PROMPT,
        prompt=prompt,
        item_count=len(candidates),
        item_label="metric",
        validator=lambda data: validate_metric_metadata(data, candidates),
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )


