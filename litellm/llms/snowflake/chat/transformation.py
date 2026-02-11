"""
Support for Snowflake REST API
"""

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Function,
    ModelResponse,
    ModelResponseStream,
)

from ...openai_like.chat.transformation import OpenAIGPTConfig

from ..utils import SnowflakeBaseConfig


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeStreamingHandler(BaseModelResponseIterator):
    """
    Custom streaming handler for Snowflake that handles missing fields in chunk responses.
    Snowflake's streaming responses may not include all OpenAI-expected fields like 'created'.
    Also transforms Claude-format tool_use to OpenAI-format tool_calls.
    """

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        # Snowflake may not include 'created' timestamp, use current time as default
        created = chunk.get("created", int(time.time()))

        # Transform choices to convert tool_use (Claude format) to tool_calls (OpenAI format)
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})

            # Check if this is a tool_use block (Claude format via Snowflake)
            if delta.get("type") == "tool_use":
                tool_call = ChatCompletionDeltaToolCall(
                    id=delta.get("tool_use_id") or delta.get("id"),
                    type="function",
                    function=Function(
                        name=delta.get("name"),
                        arguments=delta.get("input", ""),
                    ),
                    index=choice.get("index", 0),
                )
                delta["tool_calls"] = [tool_call]
                delta.pop("type", None)
                delta.pop("tool_use_id", None)
                delta.pop("content_list", None)

        return ModelResponseStream(
            id=chunk.get("id", ""),
            object="chat.completion.chunk",
            created=created,
            model=chunk.get("model", ""),
            choices=choices,
            usage=chunk.get("usage"),
        )


class SnowflakeConfig(SnowflakeBaseConfig, OpenAIGPTConfig):
    """
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api

    Snowflake Cortex LLM REST API supports function calling with specific models (e.g., Claude 3.5 Sonnet).
    This config handles transformation between OpenAI format and Snowflake's tool_spec format.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _transform_tool_calls_from_snowflake_to_openai(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """
        Transform Snowflake tool calls to OpenAI format.

        Args:
            content_list: Snowflake's content_list array containing text and tool_use items

        Returns:
            Tuple of (text_content, tool_calls)

        Snowflake format in content_list:
        {
          "type": "tool_use",
          "tool_use": {
            "tool_use_id": "tooluse_...",
            "name": "get_weather",
            "input": {"location": "Paris"}
          }
        }

        OpenAI format (returned tool_calls):
        ChatCompletionMessageToolCall(
            id="tooluse_...",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}')
        )
        """
        text_content = ""
        tool_calls: List[ChatCompletionMessageToolCall] = []

        for idx, content_item in enumerate(content_list):
            if content_item.get("type") == "text":
                text_content += content_item.get("text", "")

            ## TOOL CALLING
            elif content_item.get("type") == "tool_use":
                tool_use_data = content_item.get("tool_use", {})
                tool_call = ChatCompletionMessageToolCall(
                    id=tool_use_data.get("tool_use_id", ""),
                    type="function",
                    function=Function(
                        name=tool_use_data.get("name", ""),
                        arguments=json.dumps(tool_use_data.get("input", {})),
                    ),
                )
                tool_calls.append(tool_call)

        return text_content, tool_calls if tool_calls else None

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        response_json = raw_response.json()

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE TRANSFORMATION
        # Snowflake returns content_list (not content) with tool_use objects
        # We need to transform this to OpenAI's format with content + tool_calls
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "message" in choice and "content_list" in choice["message"]:
                content_list = choice["message"]["content_list"]
                (
                    text_content,
                    tool_calls,
                ) = self._transform_tool_calls_from_snowflake_to_openai(content_list)

                # Update the choice message with OpenAI format
                choice["message"]["content"] = text_content
                if tool_calls:
                    choice["message"]["tool_calls"] = tool_calls

                # Remove Snowflake-specific content_list
                del choice["message"]["content_list"]

        returned_response = ModelResponse(**response_json)

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model
        return returned_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Build the Snowflake Cortex inference URL.

        Handles both cases:
        - api_base is just the domain (e.g., https://account.snowflakecomputing.com)
        - api_base is the full endpoint URL (e.g., https://account.snowflakecomputing.com/api/v2/cortex/inference:complete)
        """
        endpoint = "cortex/inference:complete"

        # CARTO: skip path construction if api_base already contains the full endpoint
        # (CARTO platform may pass the full Cortex URL as api_base)
        if api_base and endpoint in api_base:
            return api_base

        api_base = self._get_api_base(api_base, optional_params)

        return f"{api_base}/{endpoint}"

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Return custom streaming handler for Snowflake that handles missing 'created' field
        and transforms Claude-format tool_use to OpenAI-format tool_calls.
        """
        return SnowflakeStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform OpenAI tool format to Snowflake tool format.

        Args:
            tools: List of tools in OpenAI format

        Returns:
            List of tools in Snowflake format

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }
        }

        Snowflake format:
        {
            "tool_spec": {
                "type": "generic",
                "name": "get_weather",
                "description": "...",
                "input_schema": {...}
            }
        }
        """
        snowflake_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                snowflake_tool: Dict[str, Any] = {
                    "tool_spec": {
                        "type": "generic",
                        "name": function.get("name"),
                        "input_schema": function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    }
                }
                # Add description if present
                if "description" in function:
                    snowflake_tool["tool_spec"]["description"] = function["description"]

                snowflake_tools.append(snowflake_tool)

        return snowflake_tools

    def _transform_tool_choice(
        self,
        tool_choice: Union[str, Dict[str, Any]],
        tool_names: Optional[List[str]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Transform OpenAI tool_choice format to Snowflake format.

        String values are converted to Snowflake's required object format:
        - "auto" -> {"type": "auto"}
        - "required" -> {"type": "required", "name": [tool_names]}
        - "none" -> {"type": "none"}

        Dict values with function type are converted:
        - {"type": "function", "function": {"name": "x"}} -> {"type": "tool", "name": ["x"]}
        """
        if isinstance(tool_choice, str):
            result: Dict[str, Any] = {"type": tool_choice}
            if tool_names and tool_choice == "required":
                result["name"] = tool_names
            return result

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_name = tool_choice.get("function", {}).get("name")
                if function_name:
                    return {
                        "type": "tool",
                        "name": [function_name],  # Snowflake expects array
                    }

        return tool_choice

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[Dict[str, Any]]:
        """
        Transform OpenAI message format to Snowflake format.

        Handles:
        - role="tool" messages -> role="user" with content_list containing tool_results
        - role="assistant" with tool_calls -> content_list with tool_use blocks
        - content=None -> content="" (Snowflake requires non-null content)
        """
        transformed_messages = []
        tool_call_map: Dict[str, str] = {}

        for message in messages:
            msg_dict = message if isinstance(message, dict) else dict(message)

            # Handle tool result messages (role="tool")
            if msg_dict.get("role") == "tool":
                tool_call_id = msg_dict.get("tool_call_id")
                content = msg_dict.get("content", "")
                tool_name = msg_dict.get("name")

                if not tool_name and tool_call_id and tool_call_id in tool_call_map:
                    tool_name = tool_call_map[tool_call_id]

                tool_results: Dict[str, Any] = {
                    "tool_use_id": tool_call_id,
                    "content": [{"type": "text", "text": str(content)}],
                }
                if tool_name:
                    tool_results["name"] = tool_name

                transformed_messages.append({
                    "role": "user",
                    "content": "",
                    "content_list": [
                        {"type": "tool_results", "tool_results": tool_results}
                    ],
                })

            # Handle assistant messages with tool_calls
            elif msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                content_list = []

                for tool_call in msg_dict.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        function_data = tool_call.get("function", {})
                        tc_id = tool_call.get("id")
                        tc_name = function_data.get("name")
                        arguments_str = function_data.get("arguments", "{}")

                        if tc_id and tc_name:
                            tool_call_map[tc_id] = tc_name

                        try:
                            arguments = (
                                json.loads(arguments_str)
                                if isinstance(arguments_str, str)
                                else arguments_str
                            )
                        except json.JSONDecodeError:
                            arguments = {}

                        content_list.append({
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": tc_id,
                                "name": tc_name,
                                "input": arguments,
                            },
                        })

                transformed_messages.append({
                    "role": "assistant",
                    "content": msg_dict.get("content") or "",
                    "content_list": content_list,
                })

            else:
                if isinstance(message, dict):
                    msg_to_append = message.copy()
                else:
                    msg_to_append = dict(message)

                content_value = msg_to_append.get("content")
                has_content_list = "content_list" in msg_to_append and msg_to_append.get("content_list")
                if content_value is None and not has_content_list:
                    msg_to_append["content"] = ""

                transformed_messages.append(msg_to_append)

        return transformed_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})

        # Transform messages to handle tool results and assistant tool_calls
        transformed_messages = self._transform_messages(messages)

        # Transform tools from OpenAI format to Snowflake's tool_spec format
        tools = optional_params.pop("tools", None)
        tool_names: List[str] = []
        if tools:
            transformed_tools = self._transform_tools(tools)
            optional_params["tools"] = transformed_tools
            tool_names = [
                t.get("tool_spec", {}).get("name")
                for t in transformed_tools
                if t.get("tool_spec", {}).get("name")
            ]

        # Transform tool_choice from OpenAI format to Snowflake's format
        tool_choice = optional_params.pop("tool_choice", None)
        if tool_choice:
            optional_params["tool_choice"] = self._transform_tool_choice(
                tool_choice, tool_names
            )

        return {
            "model": model,
            "messages": transformed_messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }
