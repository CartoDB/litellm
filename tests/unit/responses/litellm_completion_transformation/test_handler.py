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

import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD

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


_CODEX_ADDITIONAL_TOOLS_ITEM = {
    "type": "additional_tools",
    "id": "at_codex",
    "role": "developer",
    "tools": [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": "Runs a shell command.",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                },
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Waits for a background command.",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            ],
        }
    ],
}
_CODEX_INPUT = [_CODEX_ADDITIONAL_TOOLS_ITEM, {"type": "message", "role": "user", "content": "Run ls"}]


def test_sync_fallback_hoists_additional_tools_input_items_into_chat_tools():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):  # test-quality-ok: no DI seam; the file stubs this same boundary
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=False,
            )

    assert [message["role"] for message in captured["messages"]] == ["user"]
    functions_by_name = {tool["function"]["name"]: tool["function"] for tool in captured["tools"]}
    assert set(functions_by_name) == {"exec", "functions__wait"}
    assert set(functions_by_name["exec"]["parameters"]["properties"]) == {"content"}


@pytest.mark.asyncio
async def test_async_fallback_returns_hoisted_nested_custom_tool_call_as_custom_tool_call():
    from litellm.responses.litellm_completion_transformation.transformation import TOOL_CALLS_CACHE
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

    handler = LiteLLMCompletionTransformationHandler()
    tool_call_id = "call_exec_hoisted"

    async def fake_acompletion(**kwargs):
        return ModelResponse(
            id="chatcmpl-exec",
            created=1,
            model="us.openai.gpt-5.6",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(name="exec", arguments='{"content": "ls"}'),
                            )
                        ],
                    ),
                )
            ],
        )

    try:
        with patch("litellm.acompletion", fake_acompletion):  # test-quality-ok: no DI seam; file stubs this boundary
            response = await handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=True,
            )
    finally:
        TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

    tool_calls = [(item.type, item.name, item.input) for item in response.output if item.type == "custom_tool_call"]
    assert tool_calls == [("custom_tool_call", "exec", "ls")]


class _RecordingAnthropicHandler:
    def __init__(self, reply: Mapping[str, object]) -> None:
        self.reply: Final = reply
        self.request_body: Mapping[str, object] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_body = json.loads(request.content)
        return httpx.Response(200, json=dict(self.reply), request=request)


_ANTHROPIC_MESSAGE_PAYLOAD: Final = {
    "id": "msg_turn_two",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "14"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 1},
}


@pytest.mark.asyncio
async def test_bridged_follow_up_turn_keeps_the_addressed_response_id_off_the_provider_body():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        previous_response_id="resp_turn_one",
        client=client,
        **{ADDRESSED_RESPONSE_ID_FIELD: "resp_turn_one"},
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert ADDRESSED_RESPONSE_ID_FIELD not in provider.request_body, (
        f"the addressed response id reached the provider body: {sorted(provider.request_body)}"
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert [item.type for item in response.output] == ["message"]


@pytest.mark.asyncio
async def test_bridged_turn_with_a_client_field_named_previous_response_client_ids_reaches_the_provider():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        client=client,
        previous_response_client_ids=["resp_from_the_client_body"],
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_field", ["caller_input", "names_non_string_fields"])
async def test_bridged_turn_with_a_client_field_named_like_a_bridge_keyword_reaches_the_provider(client_field: str):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        client=client,
        **{client_field: "from the client body"},
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


class _StoredTurns:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [dict(row) for row in self.rows if row["session_id"] in sessions]


def _stored_mcp_turn(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "turn-one-session",
        "proxy_server_request": {"input": "My favorite number is 7.", "model": "claude-sonnet-4-6"},
        "response": {
            "id": request_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Noted.", "annotations": []}],
                },
                {"type": "mcp_tools_fetched", "tools": [{"name": "demo-echo"}]},
            ],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_under", ["litellm_encoded_id", "proxy_issued_id"])
async def test_bridged_next_turn_replays_a_turn_stored_under_the_id_the_client_holds(stored_under: str):
    litellm_encoded_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    proxy_issued_id: Final = "resp_proxy-issued-turn-one"
    stored_turns: Final = _StoredTurns(
        (_stored_mcp_turn(litellm_encoded_id if stored_under == "litellm_encoded_id" else proxy_issued_id),)
    )
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input="Double it",
            previous_response_id=litellm_encoded_id,
            client=client,
            **{ADDRESSED_RESPONSE_ID_FIELD: proxy_issued_id},
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [
        (message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]
    ] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": "Double it"}])),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_field", ["caller_input", "names_non_string_fields"])
async def test_bridged_follow_up_after_a_client_tool_search_with_a_client_field_named_like_a_bridge_keyword_reaches_the_provider(
    client_field: str,
):
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    stored_turns: Final = _StoredTurns((_stored_mcp_turn(previous_response_id),))
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        response = await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input="Double it",
            tools=[{"type": "tool_search", "execution": "client"}],
            previous_response_id=previous_response_id,
            client=client,
            **{client_field: "from the client body"},
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


class _InMemoryRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(self, name: str, value: str, ex: int) -> None:
        self.store[name] = value

    async def get(self, name: str) -> str | None:
        return self.store.get(name)


class _RedisBackedCache:
    def __init__(self, client: _InMemoryRedis) -> None:
        self.client: Final = client

    def init_async_client(self) -> _InMemoryRedis:
        return self.client


class _LiteLLMCache:
    supported_call_types: Final = ()

    def __init__(self, client: _InMemoryRedis) -> None:
        self.cache: Final = _RedisBackedCache(client)


def _recording_mock_acompletion(sent_messages: list[list[dict]], reply: str):
    real_acompletion: Final = litellm.acompletion

    async def fake_acompletion(**kwargs):
        sent_messages.append([{"role": m["role"], "content": m["content"]} for m in kwargs["messages"]])
        return await real_acompletion(**{**kwargs, "mock_response": reply})

    return fake_acompletion


async def _run_turn(user_text: str, previous_response_id: str | None, stream: bool) -> str:
    response = await litellm.aresponses(
        model="openai/gpt-4o",
        api_key="fake-api-key",
        input=user_text,
        previous_response_id=previous_response_id,
        use_chat_completions_api=True,
        stream=stream,
    )
    if not stream:
        return response.id
    completed_ids = [event.response.id async for event in response if event.type == "response.completed"]
    return completed_ids[-1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_follow_up_turn_replays_prior_turn_from_redis_session(stream: bool):
    sent_messages: list[list[dict]] = []
    redis: Final = _InMemoryRedis()

    with (
        patch("litellm.cache", _LiteLLMCache(redis)),  # test-quality-ok: session store reads the litellm.cache global
        patch(
            "litellm.acompletion", _recording_mock_acompletion(sent_messages, "Paris")
        ),  # test-quality-ok: no DI seam
    ):
        first_response_id = await _run_turn("Capital of France?", None, stream)
        await _run_turn("And its population?", first_response_id, stream)

    assert sent_messages[-1] == [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "And its population?"},
    ]


@pytest.mark.asyncio
async def test_non_streaming_session_keeps_assistant_tool_calls_for_the_follow_up_tool_result():
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

    redis: Final = _InMemoryRedis()
    tool_call: Final = ChatCompletionMessageToolCall(
        id="call_weather", type="function", function=Function(name="get_weather", arguments='{"city": "Paris"}')
    )

    async def fake_acompletion(**kwargs):
        return ModelResponse(
            id="chatcmpl-weather",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(content=None, role="assistant", tool_calls=[tool_call]),
                )
            ],
        )

    with (
        patch("litellm.cache", _LiteLLMCache(redis)),  # test-quality-ok: session store reads the litellm.cache global
        patch("litellm.acompletion", fake_acompletion),  # test-quality-ok: no DI seam; file stubs this boundary
    ):
        await _run_turn("Weather in Paris?", None, stream=False)

    stored_turns = [json.loads(session)["messages"] for session in redis.store.values()]
    assert [[(m["role"], [c["id"] for c in m.get("tool_calls", [])]) for m in turns] for turns in stored_turns] == [
        [("user", []), ("assistant", ["call_weather"])]
    ]


_TOOL_CALL_IDS: Final = ("call_a", "call_b")


def _streamed_tool_calls_acompletion(sent_messages: list[list[dict]]):
    from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function, ModelResponseStream, StreamingChoices

    real_acompletion: Final = litellm.acompletion
    chunk: Final = ModelResponseStream(
        id="chatcmpl-weather",
        choices=[
            StreamingChoices(
                index=0,
                finish_reason="tool_calls",
                delta=Delta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=call_id,
                            index=index,
                            type="function",
                            function=Function(name="get_weather", arguments='{"city": "Paris"}'),
                        )
                        for index, call_id in enumerate(_TOOL_CALL_IDS)
                    ],
                ),
            )
        ],
    )

    async def fake_acompletion(**kwargs):
        sent_messages.append(list(kwargs["messages"]))
        return await real_acompletion(**{**kwargs, "stream": True, "mock_response": chunk})

    return fake_acompletion


async def _run_streamed_turn(input: str | list[dict], previous_response_id: str | None) -> str:
    stream = await litellm.aresponses(
        model="openai/gpt-4o",
        api_key="fake-api-key",
        input=input,
        previous_response_id=previous_response_id,
        use_chat_completions_api=True,
        stream=True,
    )
    completed_ids = [event.response.id async for event in stream if event.type == "response.completed"]
    return completed_ids[-1]


def _message_shapes(messages: list[dict]) -> list[tuple[str, list[str]]]:
    return [
        (m["role"], [c["id"] for c in m.get("tool_calls", [])] or ([m["tool_call_id"]] if "tool_call_id" in m else []))
        for m in messages
    ]


@pytest.mark.asyncio
async def test_streaming_session_keeps_assistant_tool_calls_for_the_follow_up_tool_result():
    redis: Final = _InMemoryRedis()

    with (
        patch("litellm.cache", _LiteLLMCache(redis)),  # test-quality-ok: session store reads the litellm.cache global
        patch(
            "litellm.acompletion", _streamed_tool_calls_acompletion([])
        ),  # test-quality-ok: no DI seam; file stubs this boundary
    ):
        await _run_streamed_turn("Weather in Paris and Rome?", None)

    stored_turns = [json.loads(session)["messages"] for session in redis.store.values()]
    assert [_message_shapes(turns) for turns in stored_turns] == [[("user", []), ("assistant", list(_TOOL_CALL_IDS))]]


@pytest.mark.parametrize("warm_tool_calls_cache", [False, True], ids=["other_worker", "same_worker"])
@pytest.mark.asyncio
async def test_follow_up_rebuilds_the_tool_call_turn_from_the_redis_session(warm_tool_calls_cache: bool):
    from litellm.responses.litellm_completion_transformation.transformation import TOOL_CALLS_CACHE

    sent_messages: list[list[dict]] = []
    redis: Final = _InMemoryRedis()

    with (
        patch("litellm.cache", _LiteLLMCache(redis)),  # test-quality-ok: session store reads the litellm.cache global
        patch(
            "litellm.acompletion", _streamed_tool_calls_acompletion(sent_messages)
        ),  # test-quality-ok: no DI seam; file stubs this boundary
    ):
        first_response_id = await _run_streamed_turn("Weather in Paris and Rome?", None)
        if not warm_tool_calls_cache:
            for call_id in _TOOL_CALL_IDS:
                TOOL_CALLS_CACHE.delete_cache(call_id)
        await _run_streamed_turn(
            [{"type": "function_call_output", "call_id": call_id, "output": "sunny"} for call_id in _TOOL_CALL_IDS],
            first_response_id,
        )

    assert _message_shapes(sent_messages[-1]) == [
        ("user", []),
        ("assistant", list(_TOOL_CALL_IDS)),
        ("tool", ["call_a"]),
        ("tool", ["call_b"]),
    ]


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
