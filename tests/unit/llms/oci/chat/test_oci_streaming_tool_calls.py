"""
Tests for OCI streaming responses with tool calls.

This test file specifically addresses the issue where OCI API returns tool calls
without required fields like 'arguments', 'id', or 'name' during streaming,
causing Pydantic validation errors.

Issue: OCI API returns tool calls with incomplete structures during streaming
Error: ValidationError: 1 validation error for OCIStreamChunk message.toolCalls.0.arguments Field required
"""



import json
from typing import Final

import litellm
from litellm.llms.oci.chat.generic import handle_generic_stream_chunk
from litellm.llms.oci.chat.transformation import OCIStreamWrapper
from litellm.types.utils import ModelResponseStream


class TestOCIStreamingToolCalls:
    """Test cases for OCI streaming responses with incomplete tool call data."""

    def test_stream_chunk_with_missing_arguments_field(self):
        """
        OCI API can return tool calls in early chunks without the 'arguments' field,
        which should be filled with an empty string to satisfy Pydantic validation.
        """
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "name": "get_weather",
                        # 'arguments' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert len(result.choices) == 1
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_missing_id_field(self):
        """Missing 'id' gets a generated call_* id."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                        # 'id' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"].startswith("call_")

    def test_stream_chunk_with_missing_name_field(self):
        """Missing 'name' defaults to empty string."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "arguments": '{"location": "San Francisco"}',
                        # 'name' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""

    def test_stream_chunk_with_all_missing_fields(self):
        """All optional fields missing — all default gracefully."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION"
                        # id, name, arguments all missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"].startswith("call_")
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_complete_tool_call(self):
        """Fully-populated tool call passes through unchanged."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["id"] == "call_abc123"
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        )
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["arguments"]
            == '{"location": "San Francisco", "unit": "celsius"}'
        )

    def test_stream_chunk_with_multiple_tool_calls_missing_fields(self):
        """Multiple tool calls with a mix of complete and incomplete entries."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {"type": "FUNCTION", "id": "call_1", "name": "get_weather"},
                    {
                        "type": "FUNCTION",
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                    {
                        "type": "FUNCTION",
                        "id": "call_3",
                        "name": "calculate",
                        "arguments": '{"expression": "2+2"}',
                    },
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 3

        assert result.choices[0].delta.tool_calls[0]["id"] == "call_1"
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        )
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

        assert result.choices[0].delta.tool_calls[1]["id"].startswith("call_")
        assert result.choices[0].delta.tool_calls[1]["function"]["name"] == "get_time"
        assert (
            result.choices[0].delta.tool_calls[1]["function"]["arguments"]
            == '{"timezone": "UTC"}'
        )

        assert result.choices[0].delta.tool_calls[2]["id"] == "call_3"
        assert result.choices[0].delta.tool_calls[2]["function"]["name"] == "calculate"
        assert (
            result.choices[0].delta.tool_calls[2]["function"]["arguments"]
            == '{"expression": "2+2"}'
        )

    def test_stream_chunk_missing_id_is_deterministic_across_chunks(self):
        """
        Two chunks emitting the same logical tool call (same name + arguments
        at the same position) must receive the *same* synthesized id so the
        downstream stream-merger does not treat them as distinct calls.
        Random uuid4 per chunk would regress this — see bug ffdef760.
        """
        same_chunk_payload = lambda: {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                ],
            },
        }

        first = handle_generic_stream_chunk(same_chunk_payload())
        second = handle_generic_stream_chunk(same_chunk_payload())

        assert first.choices[0].delta.tool_calls is not None
        assert second.choices[0].delta.tool_calls is not None
        first_id = first.choices[0].delta.tool_calls[0]["id"]
        second_id = second.choices[0].delta.tool_calls[0]["id"]
        assert first_id == second_id
        assert first_id.startswith("call_")

    def test_stream_chunk_without_tool_calls(self):
        """Plain text chunks (no tool calls) pass through correctly."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": "Hello, how can I help you?"}],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.content == "Hello, how can I help you?"
        assert result.choices[0].delta.tool_calls is None


def _wrapper() -> OCIStreamWrapper:
    return OCIStreamWrapper(
        completion_stream=iter(()),
        model="xai.grok-4",
        logging_obj=litellm.litellm_core_utils.litellm_logging.Logging(
            model="xai.grok-4",
            messages=[],
            stream=True,
            call_type="completion",
            start_time=None,
            litellm_call_id="test",
            function_id="test",
        ),
        custom_llm_provider="oci",
    )


def _tc(call_id: str | None = None, name: str | None = None, arguments: str | None = None) -> dict[str, str]:
    return {
        "type": "FUNCTION",
        **({"id": call_id} if call_id is not None else {}),
        **({"name": name} if name is not None else {}),
        **({"arguments": arguments} if arguments is not None else {}),
    }


def _feed(wrapper: OCIStreamWrapper, chunks: list[list[dict[str, str]]]) -> list[ModelResponseStream]:
    responses: Final = [
        wrapper.chunk_creator(
            "data: " + json.dumps({"apiFormat": "GENERIC", "message": {"role": "ASSISTANT", "toolCalls": tool_calls}})
        )
        for tool_calls in chunks
    ]
    assert None not in responses
    return [r for r in responses if r is not None]


def _emit(wrapper: OCIStreamWrapper, chunks: list[list[dict[str, str]]]) -> list[tuple[int, str, str, str]]:
    return [
        (tc["index"], tc["id"], tc["function"]["name"], tc["function"]["arguments"])
        for response in _feed(wrapper, chunks)
        for tc in response.choices[0].delta.tool_calls or []
    ]


class TestOCIStreamingToolCallIdentity:
    def test_parallel_calls_in_separate_chunks_get_distinct_indices(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc("call_1", "get_weather", '{"city": "Madrid"}')],
                [_tc("call_2", "get_time", '{"tz": "UTC"}')],
            ],
        )

        assert emitted == [
            (0, "call_1", "get_weather", '{"city": "Madrid"}'),
            (1, "call_2", "get_time", '{"tz": "UTC"}'),
        ]

    def test_same_tool_called_twice_in_parallel_gets_distinct_indices(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc("call_1", "execute_query", '{"q": "a"}')],
                [_tc("call_2", "execute_query", '{"q": "b"}')],
            ],
        )

        assert emitted == [
            (0, "call_1", "execute_query", '{"q": "a"}'),
            (1, "call_2", "execute_query", '{"q": "b"}'),
        ]

    def test_interleaved_argument_fragments_follow_their_call_id(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc("call_1", "tool_a", "")],
                [_tc("call_2", "tool_b", "")],
                [_tc("call_1", "", '{"a":')],
                [_tc("call_2", "", '{"b":')],
                [_tc("call_1", "", " 1}")],
                [_tc("call_2", "", " 2}")],
            ],
        )

        assert emitted == [
            (0, "call_1", "tool_a", ""),
            (1, "call_2", "tool_b", ""),
            (0, "call_1", "", '{"a":'),
            (1, "call_2", "", '{"b":'),
            (0, "call_1", "", " 1}"),
            (1, "call_2", "", " 2}"),
        ]

    def test_two_calls_in_one_chunk_get_distinct_indices(self):
        emitted = _emit(
            _wrapper(),
            [[_tc("call_1", "tool_a", "{}"), _tc("call_2", "tool_b", "{}")]],
        )

        assert emitted == [
            (0, "call_1", "tool_a", "{}"),
            (1, "call_2", "tool_b", "{}"),
        ]

    def test_idless_call_split_across_fragments_keeps_one_index_and_id(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc(name="get_weather", arguments="")],
                [_tc(arguments='{"city": "Madrid"}')],
            ],
        )

        assert len({call_id for _, call_id, _, _ in emitted}) == 1
        assert emitted[0][1].startswith("call_")
        assert [(i, n, a) for i, _, n, a in emitted] == [
            (0, "get_weather", ""),
            (0, "", '{"city": "Madrid"}'),
        ]

    def test_late_provider_id_attaches_to_open_call_and_keeps_first_id(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc(name="get_weather", arguments="")],
                [_tc("call_late", arguments='{"city": "Madrid"}')],
            ],
        )

        assert emitted[0][1].startswith("call_")
        assert emitted[0][1] != "call_late"
        assert emitted == [
            (0, emitted[0][1], "get_weather", ""),
            (0, emitted[0][1], "", '{"city": "Madrid"}'),
        ]

    def test_idless_continuation_attaches_to_latest_open_call(self):
        emitted = _emit(
            _wrapper(),
            [
                [_tc("call_1", "tool_a", "{}")],
                [_tc("call_2", "tool_b", "")],
                [_tc(arguments='{"x": 1}')],
            ],
        )

        assert emitted == [
            (0, "call_1", "tool_a", "{}"),
            (1, "call_2", "tool_b", ""),
            (1, "call_2", "", '{"x": 1}'),
        ]

    def test_idless_complete_calls_in_separate_chunks_get_distinct_indices_and_ids(
        self,
    ):
        emitted = _emit(
            _wrapper(),
            [
                [_tc(name="get_weather", arguments='{"city": "Madrid"}')],
                [_tc(name="get_time", arguments='{"tz": "UTC"}')],
            ],
        )

        assert emitted[0][1] != emitted[1][1]
        assert [(i, n, a) for i, _, n, a in emitted] == [
            (0, "get_weather", '{"city": "Madrid"}'),
            (1, "get_time", '{"tz": "UTC"}'),
        ]

    def test_stream_chunk_builder_assembles_parallel_calls_with_emitted_ids(self):
        response_chunks = _feed(
            _wrapper(),
            [
                [_tc("call_1", "get_weather", '{"city": "Madrid"}')],
                [_tc("call_2", "get_time", '{"tz": "UTC"}')],
            ],
        )

        rebuilt = litellm.stream_chunk_builder(response_chunks)

        assert rebuilt is not None
        assert [(tc.id, tc.function.name, tc.function.arguments) for tc in rebuilt.choices[0].message.tool_calls] == [
            ("call_1", "get_weather", '{"city": "Madrid"}'),
            ("call_2", "get_time", '{"tz": "UTC"}'),
        ]

    def test_stream_chunk_builder_keeps_first_emitted_id_for_idless_split_call(self):
        response_chunks = _feed(
            _wrapper(),
            [
                [_tc(name="get_weather", arguments="")],
                [_tc(arguments='{"city": "Madrid"}')],
            ],
        )
        first_id = response_chunks[0].choices[0].delta.tool_calls[0]["id"]

        rebuilt = litellm.stream_chunk_builder(response_chunks)

        assert rebuilt is not None
        assert [(tc.id, tc.function.name, tc.function.arguments) for tc in rebuilt.choices[0].message.tool_calls] == [
            (first_id, "get_weather", '{"city": "Madrid"}')
        ]
