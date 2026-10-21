# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""The LLM model fallback: a model named in the run config (`llm_model`)
that hits its tokens-per-day quota is replaced for the rest of the run, not
asked again and again. Run with: python3 -m pytest tests"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai.inference as inference  # noqa: E402


def test_a_pinned_model_out_of_quota_is_not_asked_again(monkeypatch):
    asked = []

    def one_model(*, resolved_model, **kw):
        asked.append(resolved_model)
        if resolved_model == "big":
            raise inference._DailyQuotaExhausted("big", 1500, Exception("tokens per day"))
        return [{"ok": resolved_model}]

    monkeypatch.setattr(inference, "_EXHAUSTED_MODELS", {})
    monkeypatch.setattr(inference, "_ACTIVE_MODEL", {})
    monkeypatch.setattr(inference, "get_client", lambda provider: object())
    monkeypatch.setattr(inference, "_query_llm_json_array_one_model", one_model)
    monkeypatch.setattr(inference, "_discover_alternate_model",
                        lambda client, provider, exclude: next((m for m in ("small",) if m not in exclude), None))
    call = dict(provider="groq", model="big", system_prompt="", prompt="", item_count=1, item_label="x",
                validator=lambda x: x, retries=1, retry_delay=0, verbose=False)
    assert inference._query_llm_json_array(**call) == [{"ok": "small"}]
    assert inference._query_llm_json_array(**call) == [{"ok": "small"}]      # a later call: straight to "small"
    assert asked == ["big", "small", "small"]


def test_claude_is_a_provider(monkeypatch):
    """llm_provider: claude -- through Anthropic's OpenAI-compatible
    endpoint, with its own larger per-minute budget and an output limit."""
    config = inference.PROVIDER_CONFIG["claude"]
    assert config["base_url"] == "https://api.anthropic.com/v1/" and config["api_key_env"] == "ANTHROPIC_API_KEY"
    sent = {}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.update(kw)
                    raise inference.NotFoundError("stop", response=_Response(), body={})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(inference, "get_client", lambda provider: Client())
    monkeypatch.setattr(inference, "_discover_alternate_model", lambda *a: None)
    monkeypatch.setattr(inference, "_EXHAUSTED_MODELS", {})
    try:
        inference._query_llm_json_array(provider="claude", model=None, system_prompt="", prompt="",
                                        item_count=1, item_label="x", validator=lambda x: x, retries=1,
                                        retry_delay=0, verbose=False)
    except RuntimeError:
        pass
    assert sent["model"] == "claude-sonnet-5-5" and sent["max_tokens"] == 8192


class _Response:
    status_code = 404
    headers = {}
    request = None


def test_the_llm_is_chosen_by_name_or_by_the_key_that_is_set(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    for config in inference.PROVIDER_CONFIG.values():
        monkeypatch.delenv(config["api_key_env"], raising=False)
    assert inference.choose_provider(None) == "groq"                       # no key at all: the default
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert inference.choose_provider(None) == "openai"                     # the one whose key is set
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert inference.choose_provider(None) == "claude"                     # first in PROVIDER_ORDER
    assert inference.choose_provider("Claude") == "claude"
    assert inference.choose_provider("anthropic") == "claude"
    assert inference.choose_provider("openai") == "openai"
    import pytest
    with pytest.raises(SystemExit):
        inference.choose_provider("mistral")


def test_a_missing_key_without_a_terminal_says_how_to_give_it(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="--llm-api-key"):
        inference.get_client("openai")


def test_the_same_key_variable_for_every_llm(monkeypatch):
    for config in inference.PROVIDER_CONFIG.values():
        monkeypatch.delenv(config["api_key_env"], raising=False)
    monkeypatch.setenv("LLM_API_KEY", "the-key")
    assert [inference.api_key(p) for p in ("groq", "claude", "openai")] == ["the-key"] * 3
    assert inference.get_client("claude").api_key == "the-key"
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-2")                    # the provider's usual variable
    assert inference.api_key("claude") == "sk-ant-2" and inference.api_key("groq") is None


def test_the_key_option_and_no_keys_in_run_configs(monkeypatch, tmp_path):
    import config as run_config
    import pytest
    from metadata import builder

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    args = builder.build_arg_parser().parse_args(["/x", "--llm-api-key", "from-the-command-line"]) \
        if hasattr(builder, "build_arg_parser") else None
    if args is not None:
        assert args.llm_api_key == "from-the-command-line"
    path = tmp_path / "run.yaml"
    path.write_text("implementation: /x\ncontainer: img:1\nllm_api_key: sk-secret\n")
    with pytest.raises(SystemExit, match="don't go into run config files"):
        run_config.load_config_file(path)


def test_a_key_or_account_problem_stops_at_once_with_what_to_do(monkeypatch):
    import httpx
    import pytest

    message = ("This API key is not scoped to a workspace, so this request must include the "
               "anthropic-workspace-id header with the ID of the workspace to use.")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/chat/completions")
    error = inference.BadRequestError(message, response=httpx.Response(400, request=request),
                                      body={"type": "error", "error": {"type": "invalid_request_error",
                                                                       "message": message}})
    calls = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    raise error

    monkeypatch.setattr(inference, "get_client", lambda provider: Client())
    with pytest.raises(RuntimeError, match="key for one workspace"):
        inference._query_llm_json_array(provider="claude", model=None, system_prompt="", prompt="",
                                        item_count=1, item_label="x", validator=lambda x: x, retries=3,
                                        retry_delay=0, verbose=False)
    assert len(calls) == 1                                      # not retried


def test_the_workspace_is_sent_with_every_claude_request(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    client = inference.get_client("claude")
    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_123"


def test_claude_code_uses_the_plan_not_api_credits(monkeypatch):
    """llm_provider: claude-code asks the `claude` command, signed in with a
    Claude Pro/Max plan: no key, and API key variables are kept from it."""
    import json as _json
    import subprocess

    seen = {}

    def run(command, **kw):
        seen.update(command=command, **kw)
        reply = {"result": '[{"answer": 4}]', "is_error": False, "usage": {"input_tokens": 3, "output_tokens": 5}}
        return subprocess.CompletedProcess(command, 0, stdout=_json.dumps(reply), stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    monkeypatch.setattr(inference, "_EXHAUSTED_MODELS", {})
    assert inference.choose_provider("claude-code") == "claude-code"
    answer = inference._query_llm_json_array(provider="claude-code", model=None, system_prompt="Only JSON.",
                                             prompt="2+2?", item_count=1, item_label="x",
                                             validator=lambda x: x, retries=1, retry_delay=0, verbose=False)
    assert answer == [{"answer": 4}]
    command = seen["command"]
    assert command[:4] == ["/usr/bin/claude", "-p", "--output-format", "json"]
    assert command[command.index("--model") + 1] == "sonnet" and command[command.index("--tools") + 1] == ""
    assert seen["input"] == "2+2?" and "ANTHROPIC_API_KEY" not in seen["env"]


def test_codex_uses_the_chatgpt_plan_not_api_credits(monkeypatch):
    """llm_provider: codex asks the `codex` command, signed in with a ChatGPT
    plan: no key, and API key variables are kept from it."""
    import subprocess

    seen = {}

    def run(command, **kw):
        seen.update(command=command, **kw)
        answer = command[command.index("--output-last-message") + 1]
        Path(answer).write_text('[{"answer": 4}]')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    monkeypatch.setattr(inference, "_EXHAUSTED_MODELS", {})
    answer = inference._query_llm_json_array(provider="codex", model=None, system_prompt="Only JSON.",
                                             prompt="2+2?", item_count=1, item_label="x",
                                             validator=lambda x: x, retries=1, retry_delay=0, verbose=False)
    assert answer == [{"answer": 4}]
    command = seen["command"]
    assert command[:3] == ["/usr/bin/codex", "exec", "-"] and command[command.index("--sandbox") + 1] == "read-only"
    assert "--model" not in command                                         # Codex's own default model
    assert seen["input"] == "Only JSON.\n\n2+2?" and "OPENAI_API_KEY" not in seen["env"]
