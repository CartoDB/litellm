import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.responses.litellm_completion_transformation import session_handler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_for_previous_response_id():
    """
    Test get_chat_completion_message_history_for_previous_response_id with mock data
    """
    # Mock data based on the provided spend logs (simplified version)
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb",
            "call_type": "aresponses",
            "api_key": "sk-test-mock-api-key-123",
            "spend": 0.004803,
            "total_tokens": 329,
            "prompt_tokens": 11,
            "completion_tokens": 318,
            "startTime": "2025-05-30T03:17:06.703+00:00",
            "endTime": "2025-05-30T03:17:11.894+00:00",
            "model": "claude-sonnet-4-5-20250929",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "who is Michael Jordan",
                "model": "anthropic/claude-sonnet-4-5-20250929",
            },
            "response": {
                "id": "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb",
                "model": "claude-sonnet-4-5-20250929",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Michael Jordan (born February 17, 1963) is widely considered the greatest basketball player of all time. Here are some key points about him...",
                            "tool_calls": None,
                            "function_call": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "created": 1748575031,
                "usage": {
                    "total_tokens": 329,
                    "prompt_tokens": 11,
                    "completion_tokens": 318,
                },
            },
            "status": "success",
        },
        {
            "request_id": "chatcmpl-370760c9-39fa-4db7-b034-d1f8d933c935",
            "call_type": "aresponses",
            "api_key": "sk-test-mock-api-key-123",
            "spend": 0.010437,
            "total_tokens": 967,
            "prompt_tokens": 339,
            "completion_tokens": 628,
            "startTime": "2025-05-30T03:17:28.600+00:00",
            "endTime": "2025-05-30T03:17:39.921+00:00",
            "model": "claude-sonnet-4-5-20250929",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "can you tell me more about him",
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "previous_response_id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmFudGhyb3BpYzttb2RlbF9pZDplMGYzMDJhMTQxMmU3ODQ3MGViYjI4Y2JlZDAxZmZmNWY4OGMwZDMzMWM2NjdlOWYyYmE0YjQxM2M2ZmJkMjgyO3Jlc3BvbnNlX2lkOmNoYXRjbXBsLTkzNWI4ZGFkLWZkYzItNDY2ZS1hOGNhLWUyNmU1YThhMjFiYg==",
            },
            "response": {
                "id": "chatcmpl-370760c9-39fa-4db7-b034-d1f8d933c935",
                "model": "claude-sonnet-4-5-20250929",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Here's more detailed information about Michael Jordan...",
                            "tool_calls": None,
                            "function_call": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "created": 1748575059,
                "usage": {
                    "total_tokens": 967,
                    "prompt_tokens": 339,
                    "completion_tokens": 628,
                },
            },
            "status": "success",
        },
    ]

    # Mock the get_all_spend_logs_for_previous_response_id method
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        # Test the function
        previous_response_id = "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb"
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id
        )

        # Verify the mock was called with correct parameters
        mock_get_spend_logs.assert_called_once_with(previous_response_id)

        # Verify the returned ChatCompletionSession structure
        assert "messages" in result
        assert "litellm_session_id" in result

        # Verify session_id is extracted correctly
        assert result["litellm_session_id"] == "a96757c4-c6dc-4c76-b37e-e7dfa526b701"

        # Verify messages structure
        messages = result["messages"]
        assert len(messages) == 4  # 2 user messages + 2 assistant messages

        # Check the message sequence
        # First user message
        assert messages[0].get("role") == "user"
        assert messages[0].get("content") == "who is Michael Jordan"

        # First assistant response
        assert messages[1].get("role") == "assistant"
        content_1 = messages[1].get("content", "")
        if isinstance(content_1, str):
            assert "Michael Jordan" in content_1
            assert content_1.startswith("Michael Jordan (born February 17, 1963)")

        # Second user message
        assert messages[2].get("role") == "user"
        assert messages[2].get("content") == "can you tell me more about him"

        # Second assistant response
        assert messages[3].get("role") == "assistant"
        content_3 = messages[3].get("content", "")
        if isinstance(content_3, str):
            assert "Here's more detailed information about Michael Jordan" in content_3


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_empty_spend_logs():
    """
    Test get_chat_completion_message_history_for_previous_response_id with empty spend logs
    """
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = []

        previous_response_id = "non-existent-id"
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id
        )

        # Verify empty result structure
        assert result.get("messages") == []
        assert result.get("litellm_session_id") is None


@pytest.mark.asyncio
async def test_e2e_cold_storage_successful_retrieval():
    """
    Test end-to-end cold storage functionality with successful retrieval of full proxy request from cold storage.
    """
    # Mock spend logs with cold storage object key in metadata
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-123",
            "session_id": "session-456",
            "metadata": '{"cold_storage_object_key": "s3://test-bucket/requests/session_456_req1.json"}',
            "proxy_server_request": '{"litellm_truncated": true}',  # Truncated payload
            "response": {
                "id": "chatcmpl-test-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I am an AI assistant.",
                        },
                    }
                ],
            },
        }
    ]

    # Full proxy request data from cold storage
    full_proxy_request = {
        "input": "Hello, who are you?",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
    }

    with (
        patch.object(
            ResponsesSessionHandler,
            "get_all_spend_logs_for_previous_response_id",
            new_callable=AsyncMock,
        ) as mock_get_spend_logs,
        patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage,
        patch("litellm.cold_storage_custom_logger", return_value="s3"),
    ):

        # Setup mocks
        mock_get_spend_logs.return_value = mock_spend_logs
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = (
            AsyncMock(return_value=full_proxy_request)
        )

        # Call the main function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-123"
        )

        # Verify cold storage was called with correct object key
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key.assert_called_once_with(
            object_key="s3://test-bucket/requests/session_456_req1.json"
        )

        # Verify result structure
        assert result.get("litellm_session_id") == "session-456"
        assert len(result.get("messages", [])) >= 1  # At least the assistant response


@pytest.mark.asyncio
async def test_e2e_cold_storage_fallback_to_truncated_payload():
    """
    Test end-to-end cold storage functionality when object key is missing, falling back to truncated payload.
    """
    # Mock spend logs without cold storage object key
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-789",
            "session_id": "session-999",
            "metadata": '{"user_api_key": "test-key"}',  # No cold storage object key
            "proxy_server_request": '{"input": "Truncated message", "model": "gpt-4"}',  # Regular payload
            "response": {
                "id": "chatcmpl-test-789",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a response.",
                        },
                    }
                ],
            },
        }
    ]

    with (
        patch.object(
            ResponsesSessionHandler,
            "get_all_spend_logs_for_previous_response_id",
            new_callable=AsyncMock,
        ) as mock_get_spend_logs,
        patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage,
    ):

        # Setup mocks
        mock_get_spend_logs.return_value = mock_spend_logs

        # Call the main function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-789"
        )

        # Verify cold storage was NOT called since no object key in metadata
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key.assert_not_called()

        # Verify result structure
        assert result.get("litellm_session_id") == "session-999"
        assert len(result.get("messages", [])) >= 1  # At least the assistant response


@pytest.mark.asyncio
async def test_should_check_cold_storage_for_full_payload():
    """
    Test _should_check_cold_storage_for_full_payload returns True for proxy server requests with truncated content
    """

    # Test case 1: Proxy server request with truncated PDF content (should return True)
    proxy_request_with_truncated_pdf = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {
                        "text": "what was datadogs largest source of operating cash ? quote the section you saw ",
                        "type": "input_text",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:application/pdf;base64,JVBERi0xLjcKJYGBgYEKCjcgMCBvYmoKPDwKL0ZpbHRlciAvRmxhdGVEZWNvZGUKL0xlbmd0aCA1NjcxCj4+CnN0cmVhbQp4nO1dW4/cthV+31+h5wKVeb8AhoG9Bn0I0DYL9NlInQBFHKSpA+Tnl5qRNNRIn8ij4WpnbdqAsRaX90Oe23cOWyH94U/Dwt+/ttF/neKt59675sfPN/+9Ubp1MvwRjfAtN92fRkgn2+5jI5Xyre9++fdPN//6S/NrqCFax4XqvnVtn/631FLogjfd339+1xx/+P3nm3ffyebn/92ww2Bc46zRrGv/p5vWMOmb+N9Qb/4xtOEazn2oH3rjfV3fDTj+t6s7+zjU5XFdFxo9fPs8/CiaX26cYmc/svDjhlF+Pv7QNdT30/9wbI8dFjK0cfzhUO8wPjaOr/Eq/v/d8827vzfv37/7/v5vD6HKhw93D/c3755UI3jYuOb5p7Dsh53nYQtZqyUXugn71Dx/vnnPmHQfmuf/3HDdKhY2z8jwq8//broSjkrE/aHEtZIxZhQ/VbHHKqoVQhsv7KmKgyUWdcPkoUSHaYgwmmhk5liFt85w45WcDUC0RgntpTp1o44lMhCpl95eNOZ+AI/f3988Pp9tAV/dAu5VKz0Ls+SB0vstgNNZWTUDtw1vqC65oahKctUW5gl3etg1EnXSCWplzHewG7gDoq9jWu28DYuzPB3NucmZzm3UmvFcLI/aMpcxT0w1AvUi2aFEsJZLprxMF0xIQzmXQQArRwA2Fo/YCm7P13LxdIrodIa7QIojL5zekqblloeuwkiGI3o7jk+zMEJ9vqW+NWFDtTDnU7KtCpet1WZGHqyVxjLNxPlcdcudsU648xnNOxmIY95Wf3JduAidF3q+bgtVUC/s6VAgZ/TUE9q8oD+DCwM2qA/UFD/eWqY1RrFQ61QgUYFDBRYV3GOKkSPFaEQxQqqWWRcGzZ3u... (litellm_truncated 1197576 chars)",
                    },
                ],
            }
        ],
        "model": "anthropic/claude-4-sonnet-20250514",
        "stream": True,
        "litellm_trace_id": "16b86861-c120-4ecb-865b-4d2238bfd8f0",
    }

    # Test case 2: Regular proxy request without truncation (should return False)
    proxy_request_regular = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": "Hello, this is a regular message",
            }
        ],
        "model": "anthropic/claude-4-sonnet-20250514",
        "stream": True,
    }

    # Test case 3: Empty request (should return True)
    proxy_request_empty = {}

    # Test case 4: None request (should return True)
    proxy_request_none = None

    with patch("litellm.cold_storage_custom_logger", return_value="s3"):
        # Test case 1: Should return True for truncated content
        result1 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_with_truncated_pdf
        )
        assert (
            result1 == True
        ), "Should return True for proxy request with truncated PDF content"

        # Test case 2: Should return False for regular content
        result2 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_regular
        )
        assert (
            result2 == False
        ), "Should return False for regular proxy request without truncation"

        # Test case 3: Should return True for empty request
        result3 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_empty
        )
        assert result3 == True, "Should return True for empty proxy request"

        # Test case 4: Should return True for None request
        result4 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_none
        )
        assert result4 == True, "Should return True for None proxy request"

    # Test case 5: Should return False when cold storage is not configured
    with patch.object(litellm, "cold_storage_custom_logger", None):
        result5 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_with_truncated_pdf
        )
        assert (
            result5 == False
        ), "Should return False when cold storage is not configured, even with truncated content"


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_empty_response_dict():
    """
    Test that empty response dict is handled correctly without processing.
    This tests the fix for response validation to check for empty dict responses.
    """
    from unittest.mock import AsyncMock, patch

    # Mock spend logs with empty response dict
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-empty-response",
            "call_type": "aresponses",
            "api_key": "test_key",
            "spend": 0.001,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "startTime": "2025-01-15T10:30:00.000+00:00",
            "endTime": "2025-01-15T10:30:01.000+00:00",
            "model": "gpt-4",
            "session_id": "test-session",
            "proxy_server_request": {"input": "test input", "model": "gpt-4"},
            "response": {},  # Empty dict - should not be processed
        }
    ]

    with patch.object(
        ResponsesSessionHandler, "get_all_spend_logs_for_previous_response_id"
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        # Call the function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-empty-response"
        )

        # Verify that user message was added but no assistant response
        # Since response is empty dict, no assistant response should be processed
        # But user input from proxy_server_request should still be included
        messages = result["messages"]
        assert len(messages) == 1  # Only user message, no assistant response
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test input"

        # Verify the session was still created correctly
        assert result["litellm_session_id"] == "test-session"


@pytest.mark.asyncio
async def test_session_handler_uses_redis_first_carto_patch():
    """
    CARTO PATCH regression (PR #16): async_responses_api_session_handler must
    consult the Redis session store before the DB-backed session handler. The
    DB store is batch-written, so an immediate follow-up turn misses its own
    history without the Redis-first read. This wiring was silently dropped in
    the v1.92.0 upstream sync (helpers survived as orphans) and caused agent
    conversations to lose context and loop in integration tests.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    redis_session = {
        "messages": [
            {"role": "user", "content": "who is Michael Jordan"},
            {"role": "assistant", "content": "A basketball player."},
        ],
        "session_id": "trace-abc-123",
    }

    with patch.object(
        LiteLLMCompletionResponsesConfig,
        "_patch_get_session_from_redis",
        new=AsyncMock(return_value=redis_session),
    ) as mock_redis_get, patch.object(
        ResponsesSessionHandler,
        "get_chat_completion_message_history_for_previous_response_id",
        new=AsyncMock(),
    ) as mock_db_get:
        request = {"messages": [{"role": "user", "content": "and Scottie Pippen?"}]}
        result = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id="resp_123",
            litellm_completion_request=request,
        )

    mock_redis_get.assert_awaited_once_with("resp_123")
    mock_db_get.assert_not_awaited()
    assert result["litellm_trace_id"] == "trace-abc-123"
    assert [m["content"] for m in result["messages"]] == [
        "who is Michael Jordan",
        "A basketball player.",
        "and Scottie Pippen?",
    ]


@pytest.mark.asyncio
async def test_session_handler_falls_back_to_db_when_redis_empty():
    """CARTO PATCH: with no Redis session, the DB-backed handler is used."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    with patch.object(
        LiteLLMCompletionResponsesConfig,
        "_patch_get_session_from_redis",
        new=AsyncMock(return_value=None),
    ), patch.object(
        ResponsesSessionHandler,
        "get_chat_completion_message_history_for_previous_response_id",
        new=AsyncMock(return_value={"messages": [], "litellm_session_id": None}),
    ) as mock_db_get:
        request = {"messages": [{"role": "user", "content": "hello"}]}
        result = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id="resp_456",
            litellm_completion_request=request,
        )

    mock_db_get.assert_awaited_once()
    assert result["messages"][-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_streaming_redis_store_key_matches_decoded_lookup_key():
    """
    CARTO PATCH regression: the streaming iterator must store the Redis session
    under the DECODED response id. The response.completed event carries litellm's
    b64-encoded id, but previous_response_id is decoded (responses/utils.py)
    before it reaches the session handler - so a session stored under the encoded
    id can never be read back, and every follow-up turn loses its history.
    """
    from types import SimpleNamespace

    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    def make_chunk(content=None, finish=None):
        return ModelResponseStream(
            id="chatcmpl-real-id-123",
            choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish)],
            model="gemini-pro",
        )

    class FakeStream:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.logging_obj = SimpleNamespace(litellm_trace_id="trace-1", model_call_details={})

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._chunks:
                raise StopAsyncIteration
            return self._chunks.pop(0)

    iterator = LiteLLMCompletionStreamingIterator(
        model="gemini-pro",
        litellm_custom_stream_wrapper=FakeStream(
            [make_chunk("Hello "), make_chunk("Bristol!"), make_chunk(None, "stop")]
        ),
        request_input="Center the map on Bristol",
        responses_api_request={},
        custom_llm_provider="gemini",
        litellm_metadata={},
        litellm_completion_request={
            "messages": [{"role": "user", "content": "Center the map on Bristol"}],
            "litellm_trace_id": "trace-1",
        },
    )

    store_mock = AsyncMock()
    client_visible_id = None
    with patch.object(LiteLLMCompletionResponsesConfig, "_patch_store_session_in_redis", new=store_mock):
        async for event in iterator:
            if "completed" in str(getattr(event, "type", "")).lower():
                client_visible_id = event.response.id

    assert client_visible_id is not None
    store_mock.assert_awaited_once()
    stored_key = store_mock.await_args.kwargs["response_id"]
    lookup_key = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
        client_visible_id
    )
    assert stored_key == lookup_key
    assert store_mock.await_args.kwargs["messages"] == [
        {"role": "user", "content": "Center the map on Bristol"},
        {"role": "assistant", "content": "Hello Bristol!"},
    ]
