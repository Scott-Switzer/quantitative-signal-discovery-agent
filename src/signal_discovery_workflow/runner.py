# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Standalone plain-Python asyncio runner (decoupled from deprecated nvidia-nat).

NVIDIA deprecated NeMo Agent Toolkit (``nvidia-nat``) in October 2026 (EOL end
of 2026). This runner reads the exact same YAML config files (``llms:`` and
``workflow:``) and drives ``signal_optimizer_function`` directly using a lean
async OpenAI-compatible NIM client — zero ``nat`` CLI invocation required.

Usage:
    python -m signal_discovery_workflow.runner \
        --config_file configs/config-crsp-fundamentals.yml \
        --input "cash-flow value and insider buying signals"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class _SimpleMessage:
    content: str
    additional_kwargs: dict[str, Any]


class DirectNIMChatClient:
    """Minimal async OpenAI-compatible chat client for NVIDIA NIM endpoints."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "https://integrate.api.nvidia.com/v1/",
        temperature: float = 0.7,
        max_tokens: int = 3000,
        chat_template_kwargs: dict[str, Any] | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 4,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or ""
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))

    def _to_role_msg(self, msg: Any) -> dict[str, str]:
        if isinstance(msg, dict):
            return {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
        cls_name = type(msg).__name__.lower()
        role = "system" if "system" in cls_name else ("assistant" if "ai" in cls_name else "user")
        return {"role": role, "content": str(getattr(msg, "content", msg))}

    def _invoke_sync(self, messages: list[Any]) -> _SimpleMessage:
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set for DirectNIMChatClient")
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._to_role_msg(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs

        data = json.dumps(payload).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices") or []
                message = (choices[0].get("message") or {}) if choices else {}
                content = message.get("content") or ""
                extra_kwargs: dict[str, Any] = {"usage": body.get("usage")}
                if message.get("reasoning_content"):
                    extra_kwargs["reasoning_content"] = str(message["reasoning_content"])
                return _SimpleMessage(content=str(content), additional_kwargs=extra_kwargs)
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code not in _RETRYABLE_HTTP_CODES or attempt >= self.max_retries:
                    raise
                delay = (2.0 ** attempt) + random.uniform(0.1, 0.8)
                logger.warning(
                    f"NIM HTTP {exc.code} (attempt {attempt + 1}/{self.max_retries + 1}); "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                delay = (2.0 ** attempt) + random.uniform(0.1, 0.8)
                logger.warning(
                    f"NIM network/timeout error ({exc}, attempt {attempt + 1}/{self.max_retries + 1}); "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def ainvoke(self, messages: list[Any]) -> _SimpleMessage:
        return await asyncio.to_thread(self._invoke_sync, messages)


class DirectBuilder:
    """Drop-in replacement for NAT's Builder.get_llm()."""

    def __init__(self, llms_cfg: dict[str, dict[str, Any]]):
        self._llms_cfg = llms_cfg or {}
        self._cache: dict[str, DirectNIMChatClient] = {}

    async def get_llm(self, llm_name: str, wrapper_type: Any = None) -> DirectNIMChatClient:
        if llm_name not in self._cache:
            if llm_name not in self._llms_cfg:
                raise KeyError(f"LLM '{llm_name}' not defined in config 'llms' block")
            spec = self._llms_cfg[llm_name]
            self._cache[llm_name] = DirectNIMChatClient(
                model_name=spec["model_name"],
                base_url=spec.get("base_url", "https://integrate.api.nvidia.com/v1/"),
                temperature=spec.get("temperature", 0.7),
                max_tokens=spec.get("max_tokens", 3000),
                chat_template_kwargs=spec.get("chat_template_kwargs"),
            )
        return self._cache[llm_name]


def load_yaml_config(config_file: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse ``llms`` and ``workflow`` dicts from a standard QSD YAML config."""
    raw = yaml.safe_load(Path(config_file).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_file} must be a YAML mapping")
    llms_cfg = raw.get("llms") or {}
    wf_cfg = dict(raw.get("workflow") or {})
    wf_cfg.pop("_type", None)
    return llms_cfg, wf_cfg


async def run_workflow_direct(
    config_file: str | Path,
    request_input: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Execute the signal optimization workflow directly without the ``nat`` CLI."""
    from .signal_discovery_optimization_workflow import (
        SignalOptimizerConfig,
        signal_optimizer_function,
    )

    llms_cfg, wf_cfg = load_yaml_config(config_file)
    if overrides:
        wf_cfg.update({k: v for k, v in overrides.items() if v is not None})

    config = SignalOptimizerConfig(**wf_cfg)
    builder = DirectBuilder(llms_cfg)

    # ``@register_function`` wraps the workflow in an async context manager;
    # entering it yields the ``FunctionInfo`` (it is not an async generator).
    async with signal_optimizer_function(config, builder) as fn_info:
        fn = getattr(fn_info, "single_fn", None) or getattr(fn_info, "fn", None)
        if fn is None:
            raise RuntimeError("Could not extract runnable coroutine from FunctionInfo")
        return await fn(request_input)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lean plain-Python QSD runner (no deprecated nvidia-nat CLI)."
    )
    parser.add_argument("--config_file", "--config-file", required=True, help="Path to YAML config")
    parser.add_argument("--input", required=True, help="Research prompt / request string")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--num-signals", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    overrides: dict[str, Any] = {}
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    if args.num_signals is not None:
        overrides["num_signals"] = args.num_signals
    if overrides:
        overrides["stop_on_acceptance"] = False

    result = asyncio.run(
        run_workflow_direct(args.config_file, args.input, overrides=overrides or None)
    )
    print(result)


if __name__ == "__main__":
    main()
