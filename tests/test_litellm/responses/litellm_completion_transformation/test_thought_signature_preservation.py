"""
Tests for preserving Gemini thought_signature in Responses API session storage.

When using Gemini thinking models (2.5/3) with tool calling via the Responses API,
multi-turn conversations require the thought_signature to be preserved in the
provider_specific_fields of tool_calls.

These tests verify that:
1. _extract_assistant_message_from_response() correctly extracts tool_calls with provider_specific_fields
2. _extract_assistant_message_for_redis() in streaming iterator correctly extracts tool_calls
3. The Redis session storage includes the complete assistant response with thought_signature
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    _extract_assistant_message_from_response,
)
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


class TestExtractAssistantMessageFromResponse:
    """Tests for _extract_assistant_message_from_response in handler.py"""

    def test_extract_text_content_only(self):
        """Test extracting a simple text response without tool_calls"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello, I'm here to help!",
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert result["role"] == "assistant"
        assert result["content"] == "Hello, I'm here to help!"
        assert "tool_calls" not in result

    def test_extract_tool_calls_without_provider_specific_fields(self):
        """Test extracting tool_calls without provider_specific_fields"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_abc123"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"location": "San Francisco"}'

    def test_extract_tool_calls_with_thought_signatures_at_message_level(self):
        """
        Test extracting tool_calls with thought_signatures in message-level provider_specific_fields.
        This is how Gemini actually returns thought_signatures - at the message level, not the function level.
        """
        # Simulate a Gemini response with thought_signatures at message level
        # Gemini stores these as a list at message.provider_specific_fields.thought_signatures
        thought_signatures = [
            "VGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHRob3VnaHQgc2lnbmF0dXJl...",
            "U2Vjb25kIHRob3VnaHQgc2lnbmF0dXJl..."
        ]

        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_gemini_001",
                                "type": "function",
                                "function": {
                                    "name": "search_database",
                                    "arguments": '{"query": "user data"}',
                                },
                            }
                        ],
                        "provider_specific_fields": {
                            "thought_signatures": thought_signatures,
                        },
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["id"] == "call_gemini_001"
        assert tool_call["function"]["name"] == "search_database"

        # Critical assertion: message-level provider_specific_fields with thought_signatures must be preserved
        assert "provider_specific_fields" in result
        assert result["provider_specific_fields"]["thought_signatures"] == thought_signatures

    def test_extract_multiple_tool_calls_with_thought_signatures_at_message_level(self):
        """
        Test extracting multiple tool_calls with thought_signatures list at message level.
        Gemini provides one thought_signature per part, stored as a list at message level.
        """
        thought_signatures = ["c2lnbmF0dXJlXzE=", "c2lnbmF0dXJlXzI="]

        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "func_a",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_002",
                                "type": "function",
                                "function": {
                                    "name": "func_b",
                                    "arguments": "{}",
                                },
                            },
                        ],
                        # Gemini stores thought_signatures at message level as a list
                        "provider_specific_fields": {
                            "thought_signatures": thought_signatures,
                        },
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert len(result["tool_calls"]) == 2

        # Verify message-level thought_signatures are preserved
        assert "provider_specific_fields" in result
        assert result["provider_specific_fields"]["thought_signatures"] == thought_signatures
        assert len(result["provider_specific_fields"]["thought_signatures"]) == 2

    def test_extract_preserves_message_level_provider_specific_fields(self):
        """Test that message-level provider_specific_fields are preserved"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                        "provider_specific_fields": {
                            "safety_ratings": [{"category": "HARM_CATEGORY_DANGEROUS", "probability": "NEGLIGIBLE"}],
                        },
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert "provider_specific_fields" in result
        assert "safety_ratings" in result["provider_specific_fields"]

    def test_extract_preserves_thinking_blocks(self):
        """Test that thinking_blocks (for Claude) are preserved"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="claude-3-opus",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "Let me think about this..."}
                        ],
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert "thinking_blocks" in result
        assert result["thinking_blocks"][0]["thinking"] == "Let me think about this..."

    def test_extract_preserves_reasoning_content(self):
        """Test that reasoning_content is preserved"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning_content": "Step 1: Analyze the question...",
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert "reasoning_content" in result
        assert result["reasoning_content"] == "Step 1: Analyze the question..."

    def test_extract_returns_none_for_empty_response(self):
        """Test that None is returned when response has no choices"""
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[],
        )

        result = _extract_assistant_message_from_response(response)
        assert result is None

    def test_extract_handles_dict_tool_calls(self):
        """Test extraction when tool_calls are dict objects (not classes)"""
        # Create response with dict-based tool_calls
        response = ModelResponse(
            id="resp-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                Choices(
                    index=0,
                    finish_reason="tool_calls",
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_dict_test",
                                type="function",
                                function=Function(
                                    name="test_func",
                                    arguments='{"key": "value"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_dict_test"
        assert result["tool_calls"][0]["function"]["name"] == "test_func"


class TestStreamingIteratorExtractAssistantMessage:
    """Tests for _extract_assistant_message_for_redis in streaming_iterator.py"""

    def _create_iterator(self):
        """Helper to create a streaming iterator for testing"""
        return LiteLLMCompletionStreamingIterator(
            model="gemini-2.5-pro",
            litellm_custom_stream_wrapper=AsyncMock(),
            request_input="Test input",
            responses_api_request={},
            litellm_completion_request={"messages": []},
        )

    def test_extract_returns_none_when_no_model_response(self):
        """Test that None is returned when litellm_model_response is None"""
        iterator = self._create_iterator()
        iterator.litellm_model_response = None

        result = iterator._extract_assistant_message_for_redis()
        assert result is None

    def test_extract_tool_calls_with_thought_signatures_at_message_level(self):
        """Test that streaming iterator extracts message-level thought_signatures correctly"""
        thought_signatures = ["c3RyZWFtaW5nX3Rob3VnaHRfc2lnbmF0dXJl"]

        iterator = self._create_iterator()
        iterator.litellm_model_response = ModelResponse(
            id="resp-stream-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_stream_001",
                                "type": "function",
                                "function": {
                                    "name": "stream_func",
                                    "arguments": '{"stream": true}',
                                },
                            }
                        ],
                        # Gemini stores thought_signatures at message level
                        "provider_specific_fields": {
                            "thought_signatures": thought_signatures,
                        },
                    },
                }
            ],
        )

        result = iterator._extract_assistant_message_for_redis()

        assert result is not None
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        # Verify message-level provider_specific_fields preserved
        assert result["provider_specific_fields"]["thought_signatures"] == thought_signatures

    def test_extract_returns_none_for_empty_content_and_no_tool_calls(self):
        """Test that None is returned when there's no content and no tool_calls"""
        iterator = self._create_iterator()
        iterator.litellm_model_response = ModelResponse(
            id="resp-empty-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                    },
                }
            ],
        )

        result = iterator._extract_assistant_message_for_redis()
        assert result is None

    def test_extract_returns_message_with_content_only(self):
        """Test extraction when there's only text content"""
        iterator = self._create_iterator()
        iterator.litellm_model_response = ModelResponse(
            id="resp-text-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello from streaming!",
                    },
                }
            ],
        )

        result = iterator._extract_assistant_message_for_redis()

        assert result is not None
        assert result["role"] == "assistant"
        assert result["content"] == "Hello from streaming!"


class TestRedisSessionStorage:
    """Tests for Redis session storage integration"""

    @pytest.mark.asyncio
    async def test_non_streaming_stores_assistant_response_with_tool_calls(self):
        """
        Test that non-streaming handler stores both input messages AND assistant response
        with tool_calls and message-level provider_specific_fields in Redis.
        """
        from litellm.responses.litellm_completion_transformation.handler import (
            LiteLLMCompletionTransformationHandler,
        )
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        thought_signatures = ["dGVzdF90aG91Z2h0X3NpZ25hdHVyZQ=="]

        # Mock the completion response with tool_calls and message-level provider_specific_fields
        mock_response = ModelResponse(
            id="resp-redis-test-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_redis_001",
                                "type": "function",
                                "function": {
                                    "name": "redis_func",
                                    "arguments": '{"test": true}',
                                },
                            }
                        ],
                        # Gemini stores thought_signatures at message level
                        "provider_specific_fields": {
                            "thought_signatures": thought_signatures,
                        },
                    },
                }
            ],
        )

        stored_messages = []

        async def capture_redis_store(response_id, session_id, messages):
            stored_messages.extend(messages)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch.object(LiteLLMCompletionResponsesConfig, "transform_responses_api_request_to_chat_completion_request") as mock_transform_req, \
             patch.object(LiteLLMCompletionResponsesConfig, "transform_chat_completion_response_to_responses_api_response") as mock_transform_resp, \
             patch.object(LiteLLMCompletionResponsesConfig, "_patch_store_session_in_redis", side_effect=capture_redis_store) as mock_redis_store:

            # Setup mocks
            mock_acompletion.return_value = mock_response
            mock_transform_req.return_value = {
                "model": "gemini-2.5-pro",
                "messages": [{"role": "user", "content": "Test message"}],
            }

            # Create a mock ResponsesAPIResponse
            mock_responses_api_response = MagicMock()
            mock_responses_api_response.id = "resp_test_encoded_id"
            mock_transform_resp.return_value = mock_responses_api_response

            handler = LiteLLMCompletionTransformationHandler()
            await handler.async_response_api_handler(
                litellm_completion_request={
                    "model": "gemini-2.5-pro",
                    "messages": [{"role": "user", "content": "Test message"}],
                },
                request_input="Test message",
                responses_api_request={},
            )

            # Verify Redis store was called
            assert mock_redis_store.called

            # Verify stored messages include both user message and assistant response
            assert len(stored_messages) == 2

            # First message should be user
            assert stored_messages[0]["role"] == "user"
            assert stored_messages[0]["content"] == "Test message"

            # Second message should be assistant with tool_calls and message-level thought_signatures
            assistant_msg = stored_messages[1]
            assert assistant_msg["role"] == "assistant"
            assert "tool_calls" in assistant_msg
            assert len(assistant_msg["tool_calls"]) == 1
            # Verify message-level provider_specific_fields preserved
            assert assistant_msg["provider_specific_fields"]["thought_signatures"] == thought_signatures

    @pytest.mark.asyncio
    async def test_streaming_stores_assistant_response_with_tool_calls(self):
        """
        Test that streaming iterator stores assistant response with tool_calls
        and message-level provider_specific_fields in Redis.
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )
        from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse

        thought_signatures = ["c3RyZWFtaW5nX3Rlc3Rfc2ln"]

        # Create iterator
        iterator = LiteLLMCompletionStreamingIterator(
            model="gemini-2.5-pro",
            litellm_custom_stream_wrapper=AsyncMock(),
            request_input="Streaming test",
            responses_api_request={},
            litellm_completion_request={
                "messages": [{"role": "user", "content": "Streaming test"}]
            },
        )

        # Set up the model response with tool_calls and message-level provider_specific_fields
        iterator.litellm_model_response = ModelResponse(
            id="resp-stream-redis-1",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_stream_redis_001",
                                "type": "function",
                                "function": {
                                    "name": "stream_redis_func",
                                    "arguments": '{"stream": true}',
                                },
                            }
                        ],
                        # Gemini stores thought_signatures at message level
                        "provider_specific_fields": {
                            "thought_signatures": thought_signatures,
                        },
                    },
                }
            ],
        )

        stored_messages = []

        async def capture_redis_store(response_id, session_id, messages):
            stored_messages.extend(messages)

        # Create mock ResponseCompletedEvent
        mock_responses_api_response = MagicMock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_stream_test_id"
        mock_response_completed = ResponseCompletedEvent(
            type="response.completed",
            response=mock_responses_api_response,
        )

        with patch.object(LiteLLMCompletionResponsesConfig, "_patch_store_session_in_redis", side_effect=capture_redis_store):
            await iterator._store_session_in_redis(mock_response_completed)

            # Verify stored messages include both user message and assistant response
            assert len(stored_messages) == 2

            # First message should be user
            assert stored_messages[0]["role"] == "user"

            # Second message should be assistant with tool_calls and message-level thought_signatures
            assistant_msg = stored_messages[1]
            assert assistant_msg["role"] == "assistant"
            assert "tool_calls" in assistant_msg
            # Verify message-level provider_specific_fields preserved
            assert assistant_msg["provider_specific_fields"]["thought_signatures"] == thought_signatures


class TestLongThoughtSignaturePreservation:
    """Tests specifically for long thought_signature values that exceed typical DB limits"""

    def test_extract_preserves_long_thought_signatures_at_message_level(self):
        """
        Test that very long thought_signature values (>2048 chars) are preserved.
        This is critical because spend_tracking truncates strings >2048 chars.
        Gemini stores thought_signatures as a list at message level.
        """
        # Create a long thought_signature (simulating real Gemini signatures which can be 3000+ chars)
        long_thought_signature = "A" * 3500  # Longer than 2048 char limit

        response = ModelResponse(
            id="resp-long-sig",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_long_sig",
                                "type": "function",
                                "function": {
                                    "name": "long_sig_func",
                                    "arguments": "{}",
                                },
                            }
                        ],
                        # Gemini stores thought_signatures at message level
                        "provider_specific_fields": {
                            "thought_signatures": [long_thought_signature],
                        },
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        preserved_sig = result["provider_specific_fields"]["thought_signatures"][0]

        # The signature must be preserved in full, not truncated
        assert len(preserved_sig) == 3500
        assert preserved_sig == long_thought_signature
        assert "litellm_truncated" not in preserved_sig  # Should NOT be truncated

    def test_extract_handles_multiple_provider_specific_fields_at_message_level(self):
        """Test that multiple fields in message-level provider_specific_fields are all preserved"""
        response = ModelResponse(
            id="resp-multi-fields",
            created=123,
            model="gemini-2.5-pro",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_multi_fields",
                                "type": "function",
                                "function": {
                                    "name": "multi_fields_func",
                                    "arguments": "{}",
                                },
                            }
                        ],
                        # Gemini stores various provider-specific info at message level
                        "provider_specific_fields": {
                            "thought_signatures": ["sig123", "sig456"],
                            "safety_ratings": [{"category": "HARM", "probability": "LOW"}],
                            "model_version": "gemini-2.5-pro-preview",
                        },
                    },
                }
            ],
        )

        result = _extract_assistant_message_from_response(response)

        assert result is not None
        provider_fields = result["provider_specific_fields"]

        assert provider_fields["thought_signatures"] == ["sig123", "sig456"]
        assert provider_fields["safety_ratings"] == [{"category": "HARM", "probability": "LOW"}]
        assert provider_fields["model_version"] == "gemini-2.5-pro-preview"
