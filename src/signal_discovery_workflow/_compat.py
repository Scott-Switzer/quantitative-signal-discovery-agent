# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Optional compatibility shims for deprecated ``nvidia-nat`` and ``langchain_core``.

NVIDIA deprecated NeMo Agent Toolkit (``nvidia-nat``) in October 2026 (EOL end
of 2026). ``runner.py`` runs the workflow directly without the ``nat`` CLI.
Routing imports through this module allows the entire signal discovery stack to
import and execute cleanly even after ``nvidia-nat`` and ``langchain_core`` are
removed from the environment, while preserving full compatibility whenever they
are installed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - exercised when langchain_core is absent

    @dataclass
    class HumanMessage:  # type: ignore[no-redef]
        content: str

    @dataclass
    class SystemMessage:  # type: ignore[no-redef]
        content: str


try:
    from nat.builder.builder import Builder
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig
except ImportError:  # pragma: no cover - exercised when nvidia-nat is absent
    Builder = Any  # type: ignore[misc, assignment]

    class LLMFrameworkEnum:  # type: ignore[no-redef]
        LANGCHAIN = "langchain"

    class FunctionBaseConfig(BaseModel):  # type: ignore[no-redef]
        def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            if name is not None:
                cls._nat_name = name

    @dataclass
    class FunctionInfo:  # type: ignore[no-redef]
        single_fn: Callable[..., Any]
        description: str = ""

        @property
        def fn(self) -> Callable[..., Any]:
            return self.single_fn

        @classmethod
        def from_fn(cls, fn: Callable[..., Any], description: str = "") -> "FunctionInfo":
            return cls(single_fn=fn, description=description)

    def register_function(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # type: ignore[no-redef]
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return asynccontextmanager(fn)

        return _decorator


__all__ = [
    "Builder",
    "FunctionBaseConfig",
    "FunctionInfo",
    "HumanMessage",
    "LLMFrameworkEnum",
    "SystemMessage",
    "register_function",
]
