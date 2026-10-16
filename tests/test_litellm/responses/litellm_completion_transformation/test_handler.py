"""Regression tests for the responses -> completion fallback bridge guard.

When the Responses API falls back to chat completions (no native responses
config), it must tag the forwarded ``litellm.completion`` / ``litellm.acompletion``
call with ``_skip_responses_api_bridge=True`` so ``completion()`` does not bridge
the request straight back to the Responses API and mutually recurse forever.

Both fallback paths are covered: the sync ``response_api_handler`` (``_is_async``
False) and the async ``async_response_api_handler`` (``_is_async`` True). The
module-level ``litellm.completion`` / ``litellm.acompletion`` are patched to
capture the forwarded kwargs; if the flag-setting line is removed the captured
kwargs lack the flag and these tests fail.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)

ITERATOR_PATH = "litellm.responses.litellm_completion_transformation.handler.LiteLLMCompletionStreamingIterator"


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


def test_sync_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )

    assert captured.get("_skip_responses_api_bridge") is True


@pytest.mark.asyncio
async def test_async_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert captured.get("_skip_responses_api_bridge") is True


def _assert_iterator_got_forwarded_request(iterator_cls: MagicMock, forwarded: dict) -> None:
    passed = iterator_cls.call_args.kwargs["litellm_completion_request"]
    assert passed["messages"] == forwarded["messages"]
    assert any(m.get("role") == "user" and "hello" in str(m.get("content")) for m in passed["messages"])


def test_sync_streaming_passes_completion_request_to_iterator():
    """CARTO PATCH (PR #16): the streaming iterator stores the Redis session from this request, so without it a
    streamed turn is saved with only the assistant reply and the follow-up loses the user's messages"""
    handler = LiteLLMCompletionTransformationHandler()
    forwarded: dict = {}

    def fake_completion(**kwargs):
        forwarded.update(kwargs)
        return MagicMock(spec=litellm.CustomStreamWrapper)

    with patch("litellm.completion", fake_completion), patch(ITERATOR_PATH) as iterator_cls:
        handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            stream=True,
            _is_async=False,
        )

    _assert_iterator_got_forwarded_request(iterator_cls, forwarded)


@pytest.mark.asyncio
async def test_async_streaming_passes_completion_request_to_iterator():
    handler = LiteLLMCompletionTransformationHandler()
    forwarded: dict = {}

    async def fake_acompletion(**kwargs):
        forwarded.update(kwargs)
        return MagicMock(spec=litellm.CustomStreamWrapper)

    with patch("litellm.acompletion", fake_acompletion), patch(ITERATOR_PATH) as iterator_cls:
        await handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            stream=True,
            _is_async=True,
        )

    _assert_iterator_got_forwarded_request(iterator_cls, forwarded)
