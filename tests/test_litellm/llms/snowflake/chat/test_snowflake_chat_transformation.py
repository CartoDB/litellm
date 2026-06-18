"""
Unit tests for Snowflake chat transformation
Tests tool calling request/response transformations
"""

import os
import copy
import json

from unittest.mock import patch
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.snowflake.chat.transformation import SnowflakeConfig
from litellm.types.utils import ModelResponse


class TestSnowflakeToolTransformation:
    """Test suite for Snowflake tool calling transformations"""

    def test_transform_request_with_tools(self):
        """
        Test that OpenAI tool format is correctly transformed to Snowflake's tool_spec format.
        """
        config = SnowflakeConfig()

        # OpenAI format tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        optional_params = {"tools": tools}

        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "What's the weather?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify tools were transformed to Snowflake format
        assert "tools" in transformed_request
        assert len(transformed_request["tools"]) == 1

        snowflake_tool = transformed_request["tools"][0]
        assert "tool_spec" in snowflake_tool
        assert snowflake_tool["tool_spec"]["type"] == "generic"
        assert snowflake_tool["tool_spec"]["name"] == "get_weather"
        assert (
            snowflake_tool["tool_spec"]["description"]
            == "Get the current weather in a given location"
        )
        assert "input_schema" in snowflake_tool["tool_spec"]
        assert snowflake_tool["tool_spec"]["input_schema"]["type"] == "object"
        assert "location" in snowflake_tool["tool_spec"]["input_schema"]["properties"]

    def test_transform_request_with_tool_choice(self):
        """
        Test that OpenAI tool_choice format is correctly transformed to Snowflake format.
        """
        config = SnowflakeConfig()

        # OpenAI format tool_choice
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        optional_params = {"tool_choice": tool_choice}

        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "What's the weather?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify tool_choice was transformed to Snowflake format
        assert "tool_choice" in transformed_request
        assert transformed_request["tool_choice"]["type"] == "tool"
        assert transformed_request["tool_choice"]["name"] == [
            "get_weather"
        ]  # Array format

    def test_transform_request_with_string_tool_choice(self):
        """
        Test that string tool_choice values are converted to Snowflake's object format.
        """
        config = SnowflakeConfig()

        # "auto" and "none" become {"type": "auto"} and {"type": "none"}
        for value in ["auto", "none"]:
            optional_params = {"tool_choice": value}

            transformed_request = config.transform_request(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Test"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            assert transformed_request["tool_choice"] == {"type": value}

        # "required" becomes {"type": "required"} (no tool names without tools)
        optional_params = {"tool_choice": "required"}
        transformed_request = config.transform_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert transformed_request["tool_choice"] == {"type": "required"}

    def test_transform_response_with_tool_calls(self):
        """
        Test that Snowflake's content_list with tool_use is transformed to OpenAI format.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response with tool call
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content_list": [
                            {"type": "text", "text": ""},
                            {
                                "type": "tool_use",
                                "tool_use": {
                                    "tool_use_id": "tooluse_abc123",
                                    "name": "get_weather",
                                    "input": {
                                        "location": "Paris, France",
                                        "unit": "celsius",
                                    },
                                },
                            },
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="claude-3-5-sonnet",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # General assertions
        assert isinstance(result, ModelResponse)
        assert len(result.choices) == 1

        choice = result.choices[0]
        assert isinstance(choice, litellm.Choices)

        # Message and tool_calls assertions
        message = choice.message
        assert isinstance(message, litellm.Message)
        assert hasattr(message, "tool_calls")
        assert isinstance(message.tool_calls, list)
        assert len(message.tool_calls) == 1

        # Specific tool_call assertions
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, litellm.utils.ChatCompletionMessageToolCall)
        assert tool_call.id == "tooluse_abc123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"

        # Verify arguments are properly JSON serialized
        arguments = json.loads(tool_call.function.arguments)
        assert arguments["location"] == "Paris, France"
        assert arguments["unit"] == "celsius"

        # Verify content_list was removed and content was set
        assert message.content == ""

    def test_transform_response_with_mixed_content(self):
        """
        Test that responses with both text and tool calls are handled correctly.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response with text and tool call
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content_list": [
                            {
                                "type": "text",
                                "text": "Let me check the weather for you. ",
                            },
                            {
                                "type": "tool_use",
                                "tool_use": {
                                    "tool_use_id": "tooluse_xyz789",
                                    "name": "get_weather",
                                    "input": {"location": "Tokyo, Japan"},
                                },
                            },
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="claude-3-5-sonnet",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify text content was extracted
        message = result.choices[0].message
        assert message.content == "Let me check the weather for you. "

        # Verify tool call was also extracted
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "get_weather"

    def test_transform_response_without_tool_calls(self):
        """
        Test that regular text responses (without tools) work correctly.
        """
        config = SnowflakeConfig()

        # Mock Snowflake response without tool calls (standard response)
        mock_snowflake_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! I'm doing well, thank you for asking.",
                        "role": "assistant",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        response = httpx.Response(
            status_code=200,
            json=mock_snowflake_response,
            headers={"Content-Type": "application/json"},
        )

        model_response = ModelResponse(
            choices=[litellm.Choices(index=0, message=litellm.Message())]
        )

        logging_obj = MagicMock()

        result = config.transform_response(
            model="mistral-7b",
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify standard response works
        assert isinstance(result, ModelResponse)
        assert (
            result.choices[0].message.content
            == "Hello! I'm doing well, thank you for asking."
        )

    def test_get_supported_openai_params_includes_tools(self):
        """
        Test that tools and tool_choice are in supported params.
        """
        config = SnowflakeConfig()
        supported_params = config.get_supported_openai_params("claude-3-5-sonnet")

        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params


class TestSnowFlakeCompletion:
    model_name = "mistral"

    messages = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "the capital of France"},
    ]

    response = {
        "choices": [
            {
                "message": {
                    "content": "Paris",
                    "content_list": [{"type": "text", "text": "Paris"}],
                }
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 18, "total_tokens": 34},
    }

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_jwt_account_id(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
            api_key="00000",
            account_id="AAAA-BBBB",
        )
        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # check request
        post_kwargs = mock_post.call_args_list[-1][1]
        body = json.loads(post_kwargs["data"])
        assert body["model"] == self.model_name
        assert "the capital of France" in str(body["messages"])

        # JWT key was used
        assert "00000" in post_kwargs["headers"]["Authorization"]
        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]
        # is completion
        assert post_kwargs["url"].endswith("cortex/inference:complete")

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_pat_key_account_id(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
            api_key="pat/xxxxx",
            account_id="AAAA-BBBB",
        )
        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # PAT key was used
        post_kwargs = mock_post.call_args_list[-1][1]
        assert "xxxxx" in post_kwargs["headers"]["Authorization"]
        assert (
            post_kwargs["headers"]["X-Snowflake-Authorization-Token-Type"]
            == "PROGRAMMATIC_ACCESS_TOKEN"
        )

        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_snowflake_env(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response)

        os.environ["SNOWFLAKE_ACCOUNT_ID"] = "AAAA-BBBB"
        os.environ["SNOWFLAKE_JWT"] = "00000"

        response = litellm.completion(
            f"snowflake/{self.model_name}",
            messages=self.messages,
        )

        assert len(response.choices) == 1
        assert response.choices[0]["message"].content == "Paris"

        # JWT key was used
        post_kwargs = mock_post.call_args_list[-1][1]
        assert "00000" in post_kwargs["headers"]["Authorization"]
        # account id was used
        assert "AAAA-BBBB" in post_kwargs["url"]

        os.environ.pop("SNOWFLAKE_ACCOUNT_ID", None)
        os.environ.pop("SNOWFLAKE_JWT", None)


class TestSnowflakeAuthenticationHeaders:
    """Test suite for Snowflake authentication header handling"""

    def test_validate_environment_with_jwt(self):
        """
        Test that JWT tokens are handled correctly with KEYPAIR_JWT header.
        """
        config = SnowflakeConfig()
        headers = {}

        jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"

        result_headers = config.validate_environment(
            headers=headers,
            model="mistral-7b",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=jwt_token,
            api_base=None,
        )

        assert result_headers["Authorization"] == f"Bearer {jwt_token}"
        assert result_headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["Accept"] == "application/json"

    def test_validate_environment_with_pat_token(self):
        """
        Test that PAT tokens with pat/ prefix are handled correctly.
        The pat/ prefix should be stripped and PROGRAMMATIC_ACCESS_TOKEN should be used.
        """
        config = SnowflakeConfig()
        headers = {}

        pat_token = "pat/abc123xyz789"
        expected_token = "abc123xyz789"

        result_headers = config.validate_environment(
            headers=headers,
            model="mistral-7b",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=pat_token,
            api_base=None,
        )

        assert result_headers["Authorization"] == f"Bearer {expected_token}"
        assert result_headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["Accept"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """
        Test that missing API key raises ValueError.
        """
        config = SnowflakeConfig()
        headers = {}

        with pytest.raises(ValueError, match="Missing Snowflake JWT key"):
            config.validate_environment(
                headers=headers,
                model="mistral-7b",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )


class TestSnowflakeMessageTransformation:
    """Test suite for Snowflake message transformation (tool results and tool calls)"""

    def test_transform_messages_with_tool_result(self):
        """
        Test that OpenAI tool result messages are transformed to Snowflake format.
        """
        config = SnowflakeConfig()

        # OpenAI format with tool result
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris, France"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Temperature: 18°C, Condition: Partly cloudy",
            },
        ]

        transformed = config._transform_messages(messages)

        # First message (user) should pass through
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "What's the weather in Paris?"

        # Second message (assistant with tool_calls) should be transformed
        assert transformed[1]["role"] == "assistant"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == "Let me check the weather for you."
        assert "content_list" in transformed[1]
        assert len(transformed[1]["content_list"]) == 1  # Only tool_use, text is in content field
        assert transformed[1]["content_list"][0]["type"] == "tool_use"
        assert transformed[1]["content_list"][0]["tool_use"]["tool_use_id"] == "call_abc123"
        assert transformed[1]["content_list"][0]["tool_use"]["name"] == "get_weather"
        assert transformed[1]["content_list"][0]["tool_use"]["input"]["location"] == "Paris, France"

        # Third message (tool result) should be transformed to user message with content_list
        assert transformed[2]["role"] == "user"
        assert "content" in transformed[2]  # Must have content field
        assert transformed[2]["content"] == ""  # Empty for tool results
        assert "content_list" in transformed[2]
        assert len(transformed[2]["content_list"]) == 1
        assert transformed[2]["content_list"][0]["type"] == "tool_results"
        tool_results = transformed[2]["content_list"][0]["tool_results"]
        assert tool_results["tool_use_id"] == "call_abc123"
        assert tool_results["name"] == "get_weather"  # Name should be tracked from tool_call
        assert len(tool_results["content"]) == 1
        assert tool_results["content"][0]["type"] == "text"
        assert tool_results["content"][0]["text"] == "Temperature: 18°C, Condition: Partly cloudy"

    def test_transform_messages_regular_conversation(self):
        """
        Test that regular messages without tools pass through unchanged.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
        ]

        transformed = config._transform_messages(messages)

        # All messages should pass through unchanged
        assert len(transformed) == 3
        assert transformed[0] == messages[0]
        assert transformed[1] == messages[1]
        assert transformed[2] == messages[2]

    def test_transform_messages_assistant_without_content(self):
        """
        Test assistant messages with only tool_calls (no text content).
        """
        config = SnowflakeConfig()

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz789",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": '{"query": "users"}',
                        },
                    }
                ],
            }
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 1
        assert transformed[0]["role"] == "assistant"
        assert "content" in transformed[0]
        assert transformed[0]["content"] == ""  # Empty when no text content
        assert "content_list" in transformed[0]
        # Should only have tool_use
        assert len(transformed[0]["content_list"]) == 1
        assert transformed[0]["content_list"][0]["type"] == "tool_use"

    def test_transform_messages_multiple_tool_calls(self):
        """
        Test assistant message with multiple tool calls.
        """
        config = SnowflakeConfig()

        messages = [
            {
                "role": "assistant",
                "content": "I'll check both locations for you.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ],
            }
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 1
        assert transformed[0]["role"] == "assistant"
        assert "content" in transformed[0]
        assert transformed[0]["content"] == "I'll check both locations for you."
        assert len(transformed[0]["content_list"]) == 2  # 2 tool_use (text is in content field)
        assert transformed[0]["content_list"][0]["type"] == "tool_use"
        assert transformed[0]["content_list"][1]["type"] == "tool_use"
        assert transformed[0]["content_list"][0]["tool_use"]["tool_use_id"] == "call_1"
        assert transformed[0]["content_list"][1]["tool_use"]["tool_use_id"] == "call_2"

    def test_transform_request_integrates_message_transformation(self):
        """
        Test that transform_request properly calls _transform_messages.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "tool",
                "tool_call_id": "call_test",
                "content": "Tool result",
            },
        ]

        result = config.transform_request(
            model="claude-3-5-sonnet",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Check that messages were transformed
        assert "messages" in result
        assert len(result["messages"]) == 2
        # Second message should be transformed from tool to user with content_list
        assert result["messages"][1]["role"] == "user"
        assert "content" in result["messages"][1]  # Must have content field
        assert "content_list" in result["messages"][1]

    def test_transform_messages_with_none_content(self):
        """
        Test that messages with content=None are normalized to have content=""
        This is critical for previous_response_id scenarios where loaded messages
        may have None content.
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},  # content=None (no tool_calls)
            {"role": "assistant"},  # Missing content field entirely
        ]

        transformed = config._transform_messages(messages)

        # All messages should have content field, even if it was None or missing
        assert len(transformed) == 3
        assert transformed[0]["content"] == "Hello"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == ""  # None should become empty string
        assert "content" in transformed[2]
        assert transformed[2]["content"] == ""  # Missing should become empty string

    def test_transform_messages_preserves_existing_content(self):
        """
        Test that messages with valid content are not modified
        """
        config = SnowflakeConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I can help with that."},
        ]

        transformed = config._transform_messages(messages)

        assert len(transformed) == 3
        assert transformed[0]["content"] == "You are a helpful assistant."
        assert transformed[1]["content"] == "What's the weather?"
        assert transformed[2]["content"] == "I can help with that."

    def test_transform_messages_previous_response_id_scenario(self):
        """
        Test the exact scenario that causes the error:
        - First request returns assistant message with tool_calls (content may be None)
        - Message is loaded from spend logs without tool_calls field
        - Should be normalized to have content field
        """
        config = SnowflakeConfig()

        # Simulate messages loaded from previous_response_id
        # The assistant message from the first response no longer has tool_calls
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,  # Was returned from Snowflake with no text, only tool_calls
                # Note: tool_calls field is NOT present because it was consumed
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Temperature: 18°C",
            },
        ]

        transformed = config._transform_messages(messages)

        # All messages should be properly formatted
        assert len(transformed) == 3
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "What's the weather in Paris?"

        # The problematic assistant message should now have content field
        assert transformed[1]["role"] == "assistant"
        assert "content" in transformed[1]
        assert transformed[1]["content"] == ""  # Normalized from None

        # Tool result should be transformed properly
        assert transformed[2]["role"] == "user"
        assert "content" in transformed[2]
        assert transformed[2]["content"] == ""


class TestSnowflakeStreamingHandler:
    """Test suite for Snowflake streaming response handling"""

    def test_chunk_parser_with_created_field(self):
        """
        Test that streaming chunks with 'created' field are parsed correctly.
        This is the standard case for models like mistral-7b and llama3.3.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
        )

        handler = SnowflakeStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = {
            "id": "chatcmpl-123",
            "created": 1234567890,
            "model": "mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-123"
        assert result.created == 1234567890
        assert result.model == "mistral-7b"
        assert result.object == "chat.completion.chunk"
        assert len(result.choices) == 1

    def test_chunk_parser_without_created_field(self):
        """
        Test that streaming chunks WITHOUT 'created' field are parsed correctly.
        This handles the case for Claude models (sonnet-3.5, sonnet-4-5) which
        don't include the 'created' field in their streaming responses.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
        )

        handler = SnowflakeStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Chunk without 'created' field (like claude-sonnet-4-5)
        chunk = {
            "id": "chatcmpl-456",
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hi there"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-456"
        assert result.created is not None  # Should have a default timestamp
        assert isinstance(result.created, int)  # Should be an integer timestamp
        assert result.model == "claude-sonnet-4-5"
        assert result.object == "chat.completion.chunk"
        assert len(result.choices) == 1

    def test_get_model_response_iterator(self):
        """
        Test that SnowflakeConfig returns the custom streaming handler.
        """
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
            SnowflakeConfig,
        )

        config = SnowflakeConfig()

        handler = config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        assert isinstance(handler, SnowflakeStreamingHandler)


class TestSnowflakeCortex390142Fixes:
    """
    Regression coverage for sc-554963 — Snowflake Cortex `390142 Incoming
    request does not contain a valid payload` on Claude function-calling
    follow-up turns.

    Three root causes (mirrors the Databricks PR #110 family + one
    Snowflake-specific schema fix):

    1. Streaming tool_use chunks emit `input=""` for parameterless calls
       (Bedrock-Anthropic quirk) → SDK accumulates `""` → invalid JSON →
       poisoned conversation history. Fix: seed `"{}"` on the
       name-introducing chunk.
    2. OpenAI-compatible providers emit `annotations: []` inside text
       content blocks; Snowflake rejects unknown fields. Fix: strip.
    3. The Agents SDK replays a tool-using turn as TWO consecutive
       assistant messages (text-only, then tool_call-only). Snowflake
       expects one assistant message per turn. Fix: merge.
    """

    # ----- chunk_parser: empty-args seeding ----------------------------

    def _make_handler(self):
        from litellm.llms.snowflake.chat.transformation import (
            SnowflakeStreamingHandler,
        )

        return SnowflakeStreamingHandler.__new__(SnowflakeStreamingHandler)

    def _tool_use_chunk(self, **delta_overrides):
        delta = {"type": "tool_use", "tool_use_id": "tu1"}
        delta.update(delta_overrides)
        return {
            "id": "chunk-1",
            "model": "claude-3-5-sonnet",
            "created": 1,
            "choices": [{"index": 0, "delta": delta}],
        }

    def test_chunk_parser_seeds_empty_string_input_with_braces(self):
        handler = self._make_handler()

        result = handler.chunk_parser(
            self._tool_use_chunk(name="get_coords", input="")
        )

        tc = result.choices[0].delta.tool_calls[0]
        assert tc.function.arguments == "{}"
        assert tc.function.name == "get_coords"

    def test_chunk_parser_seeds_missing_input_with_braces(self):
        handler = self._make_handler()

        chunk = self._tool_use_chunk(name="get_coords")
        # No `input` key at all.
        result = handler.chunk_parser(chunk)

        tc = result.choices[0].delta.tool_calls[0]
        assert tc.function.arguments == "{}"

    def test_chunk_parser_serializes_dict_input_to_json_string(self):
        handler = self._make_handler()

        result = handler.chunk_parser(
            self._tool_use_chunk(
                name="wx", input={"location": "Madrid", "unit": "celsius"}
            )
        )

        tc = result.choices[0].delta.tool_calls[0]
        # ChatCompletionDeltaToolCall.arguments must be a string for the
        # downstream SDK to accumulate correctly.
        assert isinstance(tc.function.arguments, str)
        parsed = json.loads(tc.function.arguments)
        assert parsed == {"location": "Madrid", "unit": "celsius"}

    def test_chunk_parser_passes_through_partial_json_string(self):
        """
        Subsequent delta chunks (no name, partial JSON in `input`) must
        pass through untouched so consumers can accumulate them.
        """
        handler = self._make_handler()

        result = handler.chunk_parser(
            self._tool_use_chunk(name=None, input='{"loc')
        )

        tc = result.choices[0].delta.tool_calls[0]
        assert tc.function.arguments == '{"loc'

    def test_chunk_parser_does_not_seed_when_no_name(self):
        """
        Empty `input` on a continuation chunk (no name) must not be
        rewritten — only the name-introducing chunk seeds.
        """
        handler = self._make_handler()

        result = handler.chunk_parser(
            self._tool_use_chunk(name=None, input="")
        )

        tc = result.choices[0].delta.tool_calls[0]
        assert tc.function.arguments == ""

    def test_chunk_parser_single_tool_call_starts_at_index_zero(self):
        handler = self._make_handler()

        result = handler.chunk_parser(self._tool_use_chunk(name="get_coords"))

        assert result.choices[0].delta.tool_calls[0].index == 0

    def test_chunk_parser_assigns_distinct_index_per_tool_call(self):
        """
        Cortex streams every tool_use block on the single choice (index 0).
        Each name-introducing chunk starts a new tool call and must get a
        distinct, monotonic index; otherwise the downstream accumulator
        concatenates the names/arguments of separate calls into one
        malformed tool call (e.g. `set_layer_style` +
        `set_map_center_and_zoom_to_layer`).
        """
        handler = self._make_handler()

        first = handler.chunk_parser(
            self._tool_use_chunk(name="set_layer_style", input={})
        )
        second = handler.chunk_parser(
            self._tool_use_chunk(name="set_map_center_and_zoom_to_layer", input={})
        )

        assert first.choices[0].delta.tool_calls[0].index == 0
        assert second.choices[0].delta.tool_calls[0].index == 1

    def test_chunk_parser_continuation_chunks_keep_current_index(self):
        """
        Continuation chunks (name=None, partial JSON arguments) must reuse
        the index of the tool call whose arguments they are streaming, so
        the accumulator appends to the right call.
        """
        handler = self._make_handler()

        handler.chunk_parser(self._tool_use_chunk(name="first_tool", input=""))
        cont1 = handler.chunk_parser(self._tool_use_chunk(name=None, input='{"a'))
        handler.chunk_parser(self._tool_use_chunk(name="second_tool", input=""))
        cont2 = handler.chunk_parser(self._tool_use_chunk(name=None, input='{"b'))

        assert cont1.choices[0].delta.tool_calls[0].index == 0
        assert cont2.choices[0].delta.tool_calls[0].index == 1

    def test_chunk_parser_strips_snowflake_specific_delta_fields(self):
        handler = self._make_handler()

        result = handler.chunk_parser(
            self._tool_use_chunk(name="get_coords", input="")
        )

        delta = result.choices[0].delta
        # `type`, `tool_use_id`, `input`, `name`, and `content_list`
        # should be removed in favour of the OpenAI-shaped `tool_calls`.
        for snowflake_field in ("type", "tool_use_id", "input", "name", "content_list"):
            assert not hasattr(delta, snowflake_field) or getattr(delta, snowflake_field) is None

    # ----- _strip_openai_annotations ----------------------------------

    def test_strip_openai_annotations_removes_field(self):
        from litellm.llms.snowflake.chat.transformation import (
            _strip_openai_annotations,
        )

        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi", "annotations": []},
                {"type": "text", "text": "there", "annotations": [{"x": 1}]},
            ],
        }

        _strip_openai_annotations(message)

        assert "annotations" not in message["content"][0]
        assert "annotations" not in message["content"][1]
        # Other fields preserved.
        assert message["content"][0] == {"type": "text", "text": "hi"}

    def test_strip_openai_annotations_is_noop_on_string_content(self):
        from litellm.llms.snowflake.chat.transformation import (
            _strip_openai_annotations,
        )

        message = {"role": "user", "content": "hello"}
        _strip_openai_annotations(message)
        assert message == {"role": "user", "content": "hello"}

    def test_strip_openai_annotations_is_noop_on_none_content(self):
        from litellm.llms.snowflake.chat.transformation import (
            _strip_openai_annotations,
        )

        message = {"role": "assistant", "content": None}
        _strip_openai_annotations(message)
        assert message == {"role": "assistant", "content": None}

    # ----- _content_to_text_blocks ------------------------------------

    def test_content_to_text_blocks_from_string(self):
        from litellm.llms.snowflake.chat.transformation import (
            _content_to_text_blocks,
        )

        assert _content_to_text_blocks("hello") == [
            {"type": "text", "text": "hello"}
        ]

    def test_content_to_text_blocks_drops_empty_and_non_text(self):
        from litellm.llms.snowflake.chat.transformation import (
            _content_to_text_blocks,
        )

        blocks = _content_to_text_blocks(
            [
                {"type": "text", "text": "a", "annotations": []},
                {"type": "text", "text": "   "},  # whitespace-only → drop
                {"type": "tool_use", "tool_use": {}},  # non-text → drop
                {"type": "text", "text": "b"},
            ]
        )

        assert blocks == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_content_to_text_blocks_handles_empty_inputs(self):
        from litellm.llms.snowflake.chat.transformation import (
            _content_to_text_blocks,
        )

        assert _content_to_text_blocks("") == []
        assert _content_to_text_blocks(None) == []
        assert _content_to_text_blocks([]) == []
        assert _content_to_text_blocks("   ") == []

    # ----- _transform_messages: consecutive-assistant merge -----------

    def test_transform_messages_merges_consecutive_assistant_messages(self):
        """
        Reproduces the captured failing payload shape from sc-554963:
        an assistant text message immediately followed by an assistant
        tool_call message. After transform they should be a single
        assistant message with content_list containing the text block
        first and the tool_use block second.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Call get_map_coordinates."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll call the tool.",
                            "annotations": [],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "toolu_bdrk_01",
                            "type": "function",
                            "function": {
                                "name": "get_map_coordinates",
                                "arguments": "",
                            },
                        }
                    ],
                },
            ]
        )

        # user + ONE merged assistant
        assert len(transformed) == 2
        assert transformed[0]["role"] == "user"

        assistant = transformed[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == ""
        assert len(assistant["content_list"]) == 2

        text_block, tool_block = assistant["content_list"]
        # Text prelude carried over (and annotations stripped).
        assert text_block == {"type": "text", "text": "I'll call the tool."}
        # Tool_use built from the second message with empty args coerced
        # to {} (parameterless call).
        assert tool_block["type"] == "tool_use"
        assert tool_block["tool_use"]["tool_use_id"] == "toolu_bdrk_01"
        assert tool_block["tool_use"]["name"] == "get_map_coordinates"
        assert tool_block["tool_use"]["input"] == {}

    def test_transform_messages_does_not_merge_when_previous_has_content_list(self):
        """
        A previous assistant that already carried tool_calls should not
        be merged with the next assistant — they belong to different
        turns separated by a tool_result.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "First."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu1",
                            "type": "function",
                            "function": {"name": "first", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tu1", "content": "ok"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu2",
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )

        # user + assistant + user(tool_results) + assistant — no merge.
        assert len(transformed) == 4
        assert transformed[1]["role"] == "assistant"
        assert transformed[2]["role"] == "user"
        assert transformed[3]["role"] == "assistant"

    def test_transform_messages_does_not_merge_across_non_assistant(self):
        """
        If a non-assistant message sits between two assistants, merging
        must not happen.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "First."},
                {"role": "assistant", "content": "I think..."},
                {"role": "user", "content": "Now do something."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu1",
                            "type": "function",
                            "function": {"name": "do_it", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )

        assert len(transformed) == 4
        # The first assistant remains a plain text message.
        assert transformed[1]["role"] == "assistant"
        assert transformed[1].get("content_list") in (None, [])
        # The second assistant has just the tool_use block, no prelude.
        last = transformed[3]
        assert len(last["content_list"]) == 1
        assert last["content_list"][0]["type"] == "tool_use"

    def test_transform_messages_strips_annotations_on_plain_assistant(self):
        """
        Annotations must be stripped from plain (non-merged) assistant
        text messages as well — Cortex rejects them with 390142
        regardless of merge.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello there.",
                            "annotations": [],
                        }
                    ],
                },
            ]
        )

        assert "annotations" not in transformed[1]["content"][0]

    def test_transform_messages_empty_tool_call_arguments_coerced_to_empty_dict(self):
        """
        Without a merge in play, an assistant message with `tool_calls`
        whose `arguments=""` must still serialize to `input={}` in the
        Snowflake tool_use block — Snowflake's schema requires an object
        for `input`, not an empty string.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Call it."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu1",
                            "type": "function",
                            "function": {
                                "name": "get_coords",
                                "arguments": "",
                            },
                        }
                    ],
                },
            ]
        )

        tool_use = transformed[1]["content_list"][0]
        assert tool_use["tool_use"]["input"] == {}

    def test_transform_messages_missing_tool_call_arguments_coerced_to_empty_dict(self):
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Call it."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu1",
                            "type": "function",
                            "function": {"name": "get_coords"},
                            # no `arguments` key at all
                        }
                    ],
                },
            ]
        )

        tool_use = transformed[1]["content_list"][0]
        assert tool_use["tool_use"]["input"] == {}

    def test_transform_messages_preserves_non_empty_arguments(self):
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Get the weather."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        }
                    ],
                },
            ]
        )

        tool_use = transformed[1]["content_list"][0]
        assert tool_use["tool_use"]["input"] == {"location": "Paris"}

    def test_transform_messages_end_to_end_captured_failing_payload(self):
        """
        Full end-to-end smoke test against the exact shape captured from
        the sc-554963 production HAR (annotations + two-consecutive
        assistants + empty-string arguments + synthetic tool_result).

        After the fixes the Snowflake-bound payload must:
        - have a SINGLE assistant message per turn (no consecutive
          same-role messages),
        - carry NO `annotations` fields anywhere in content blocks,
        - have `tool_use.input` as `{}` (object), not `""`.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "system", "content": "You are an assistant."},
                {
                    "role": "user",
                    "content": "Call get_map_coordinates and echo the result.",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll call the tool and share the result.",
                            "annotations": [],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "toolu_bdrk_01U31Y2KLfXpuoeg2f2DqgcU",
                            "type": "function",
                            "function": {
                                "name": "get_map_coordinates",
                                "arguments": "",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_bdrk_01U31Y2KLfXpuoeg2f2DqgcU",
                    "name": "get_map_coordinates",
                    "content": '{"latitude": 40.4168, "longitude": -3.7038}',
                },
            ]
        )

        # Expected shape: system, user, merged-assistant, user(tool_results)
        assert [m["role"] for m in transformed] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

        # No consecutive same-role messages.
        for i in range(len(transformed) - 1):
            assert transformed[i]["role"] != transformed[i + 1]["role"]

        # The merged assistant: text prelude + tool_use, no annotations
        # leaked anywhere, input is an object.
        merged = transformed[2]
        assert len(merged["content_list"]) == 2
        text_block, tool_block = merged["content_list"]
        assert text_block == {
            "type": "text",
            "text": "I'll call the tool and share the result.",
        }
        assert tool_block["tool_use"]["input"] == {}
        assert tool_block["tool_use"]["name"] == "get_map_coordinates"

        # tool_result wrapped into user message with content_list.
        tool_user = transformed[3]
        assert tool_user["content_list"][0]["type"] == "tool_results"
        assert (
            tool_user["content_list"][0]["tool_results"]["tool_use_id"]
            == "toolu_bdrk_01U31Y2KLfXpuoeg2f2DqgcU"
        )

        # Recursively assert no `annotations` field survives anywhere in
        # the transformed payload — this is the field Snowflake rejects.
        def _no_annotations(obj):
            if isinstance(obj, dict):
                assert "annotations" not in obj, f"annotations leaked in {obj}"
                for v in obj.values():
                    _no_annotations(v)
            elif isinstance(obj, list):
                for item in obj:
                    _no_annotations(item)

        _no_annotations(transformed)


class TestSnowflakeCortexArrayContentFlatten:
    """
    Regression coverage for the sc-554963 follow-up: Snowflake Cortex
    `inference:complete` rejects a message whose `content` is an ARRAY of
    content blocks (`[{"type": "text", "text": "..."}]`) with
    `390142 Incoming request does not contain a valid payload`. It requires
    `content` to be a plain string.

    The OpenAI Agents SDK replays a prior assistant turn with exactly that
    list-of-blocks shape on every follow-up turn, so a PLAIN multi-turn text
    conversation (no tools at all) fails on the second turn. This was missed
    by the original fix because:
      - the only end-to-end repro covered the tool-call path, never a plain
        text multi-turn, and
      - the unit tests exercised list-form content only in the merge path
        (where it is converted into content_list), never on a standalone
        assistant message that goes through the passthrough branch.

    Verified against the live Cortex endpoint: array-form content -> HTTP 400
    / 390142; the same payload flattened to a string -> HTTP 200.
    """

    def test_helper_flattens_list_to_string(self):
        from litellm.llms.snowflake.chat.transformation import (
            _content_to_text_string,
        )

        assert _content_to_text_string("hello") == "hello"
        assert _content_to_text_string(None) == ""
        assert _content_to_text_string([]) == ""
        assert (
            _content_to_text_string(
                [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
            )
            == "ab"
        )
        # non-text blocks are ignored
        assert (
            _content_to_text_string(
                [{"type": "text", "text": "a"}, {"type": "image_url"}]
            )
            == "a"
        )

    def test_standalone_assistant_array_content_flattened_to_string(self):
        """
        The exact production-failing shape from the capture:
        user "Hi" -> assistant replayed with list-form content -> follow-up
        user message. The assistant content must come out as a string, and
        the message must NOT acquire a content_list.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hi! How can I help you with the cell towers map today?",
                        }
                    ],
                },
                {"role": "user", "content": "Center the map in Cuenca"},
            ]
        )

        assistant = transformed[2]
        assert assistant["role"] == "assistant"
        assert isinstance(assistant["content"], str)
        assert (
            assistant["content"]
            == "Hi! How can I help you with the cell towers map today?"
        )
        # A plain assistant turn must not gain a content_list.
        assert "content_list" not in assistant

    def test_no_message_has_array_form_content_after_transform(self):
        """
        Guard: after transform, NO message may carry list-form `content`
        (the shape Cortex rejects). Mixes plain text turns and a tool turn.
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                },
                {"role": "user", "content": "do it"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "calling"}],
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )

        for m in transformed:
            assert not isinstance(
                m.get("content"), list
            ), f"message still has array-form content: {m}"

    def test_tool_call_assistant_array_content_flattened(self):
        """
        An assistant message that carries BOTH list-form text content and
        tool_calls must emit string `content` (the tool_use goes in
        content_list).
        """
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "do it"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "on it"}],
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "f", "arguments": ""},
                        }
                    ],
                },
            ]
        )

        assistant = transformed[-1]
        assert isinstance(assistant["content"], str)
        assert assistant["content"] == "on it"
        assert assistant["content_list"][-1]["type"] == "tool_use"
        assert assistant["content_list"][-1]["tool_use"]["input"] == {}

    def test_system_message_array_content_flattened(self):
        """System messages can also arrive with list-form content."""
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "line1 "},
                        {"type": "text", "text": "line2"},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ]
        )

        assert transformed[0]["role"] == "system"
        assert transformed[0]["content"] == "line1 line2"

    def test_empty_array_content_becomes_empty_string(self):
        config = SnowflakeConfig()

        transformed = config._transform_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": []},
            ]
        )

        assistant = transformed[-1]
        assert assistant["content"] == ""
        assert "content_list" not in assistant
