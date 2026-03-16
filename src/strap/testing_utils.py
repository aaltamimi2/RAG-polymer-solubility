"""Shared helpers for deterministic tests and harnesses."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch


def blocked_model_call(counter: Counter[str] | None, label: str):
    """Return a callable that raises when a real model path is touched."""

    def _raiser(*args: Any, **kwargs: Any):
        if counter is not None:
            counter[label] += 1
        raise AssertionError(f"Blocked model/Gemini access via {label}")

    return _raiser


def block_model_access(counter: Counter[str] | None = None) -> ExitStack:
    """Patch common model entry points so deterministic tests stay offline."""
    stack = ExitStack()

    try:
        import langchain.chat_models as chat_models
    except Exception:  # pragma: no cover
        chat_models = None
    if chat_models is not None and hasattr(chat_models, "init_chat_model"):
        stack.enter_context(
            patch.object(
                chat_models,
                "init_chat_model",
                side_effect=blocked_model_call(counter, "langchain.chat_models.init_chat_model"),
            )
        )

    try:
        from langchain_core.language_models.chat_models import BaseChatModel
    except Exception:  # pragma: no cover
        BaseChatModel = None
    if BaseChatModel is not None:
        stack.enter_context(
            patch.object(
                BaseChatModel,
                "invoke",
                side_effect=blocked_model_call(counter, "BaseChatModel.invoke"),
            )
        )
        stack.enter_context(
            patch.object(
                BaseChatModel,
                "ainvoke",
                side_effect=blocked_model_call(counter, "BaseChatModel.ainvoke"),
            )
        )

    try:
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    except Exception:  # pragma: no cover
        ChatGoogleGenerativeAI = None
    if ChatGoogleGenerativeAI is not None:
        stack.enter_context(
            patch.object(
                ChatGoogleGenerativeAI,
                "invoke",
                side_effect=blocked_model_call(counter, "ChatGoogleGenerativeAI.invoke"),
            )
        )
        stack.enter_context(
            patch.object(
                ChatGoogleGenerativeAI,
                "ainvoke",
                side_effect=blocked_model_call(counter, "ChatGoogleGenerativeAI.ainvoke"),
            )
        )

    try:
        from google.genai.models import AsyncModels, Models
    except Exception:  # pragma: no cover
        Models = None
        AsyncModels = None
    if Models is not None:
        for attr in ("generate_content", "generate_content_stream"):
            if hasattr(Models, attr):
                stack.enter_context(
                    patch.object(
                        Models,
                        attr,
                        side_effect=blocked_model_call(counter, f"google.genai.models.Models.{attr}"),
                    )
                )
    if AsyncModels is not None:
        for attr in ("generate_content", "generate_content_stream"):
            if hasattr(AsyncModels, attr):
                stack.enter_context(
                    patch.object(
                        AsyncModels,
                        attr,
                        side_effect=blocked_model_call(counter, f"google.genai.models.AsyncModels.{attr}"),
                    )
                )

    return stack
