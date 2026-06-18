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


def _strip_openai_annotations(message_dict: Dict[str, Any]) -> None:
    """
    Remove the OpenAI-only `annotations` field from each content block.
    Snowflake Cortex rejects unknown fields inside text content blocks with
    `390142 Incoming request does not contain a valid payload`. The field
    appears on assistant messages emitted by OpenAI-compatible providers
    (for citation parity with the Responses API) and survives the Agents
    SDK replay on every follow-up turn.
    """
    content = message_dict.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and "annotations" in block:
            block.pop("annotations", None)


def _content_to_text_blocks(content: Any) -> List[Dict[str, Any]]:
    """
    Convert an OpenAI-style assistant `content` value into a list of Snowflake
    `{"type": "text", "text": ...}` blocks suitable for inclusion in
    `content_list`. Empty/whitespace-only text is dropped.
    """
    if not content:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text") or ""
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
        return blocks
    return []


def _content_to_text_string(content: Any) -> str:
    """
    Flatten an OpenAI-style `content` value into a plain string.

    Snowflake Cortex `inference:complete` rejects a message whose `content`
    is an array of content blocks (e.g. `[{"type": "text", "text": "..."}]`)
    with `390142 Incoming request does not contain a valid payload`; it
    requires `content` to be a plain string. The OpenAI Agents SDK replays a
    prior assistant turn with exactly that list-of-blocks shape on every
    follow-up turn, so a plain multi-turn text conversation (no tools
    involved) hits 390142 on the second turn. Concatenating the text blocks
    back into a string is the form Cortex accepts. Non-text blocks are
    ignored here — tool_use / tool_results are carried in `content_list`.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


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
                name = delta.get("name")

                # Assign a distinct, monotonic index per tool call. Cortex
                # streams every tool_use block on the single choice (index 0),
                # so using `choice.index` collapses multiple tool calls onto
                # one index — the downstream accumulator then concatenates
                # their names and arguments into a single malformed call
                # (e.g. `set_layer_style` + `set_map_center_and_zoom_to_layer`).
                # A new tool call always starts with a name-introducing chunk;
                # continuation chunks (name=None) stream the arguments and must
                # keep the current index.
                tool_call_index = getattr(self, "_tool_call_index", -1)
                if name:
                    tool_call_index += 1
                    self._tool_call_index = tool_call_index

                # Normalize `input` into a JSON string for the OpenAI delta shape.
                # Cortex routes Claude through Bedrock, which emits parameterless
                # tool_use chunks with `input=""` instead of `input={}`. The SDK
                # would accumulate the empty string into an invalid arguments
                # value (JSON.parse fails on ""), which then poisons the
                # conversation history on the next turn. Seed `"{}"` on the
                # name-introducing chunk so accumulation ends up valid JSON.
                input_value = delta.get("input")
                if isinstance(input_value, dict):
                    arguments = json.dumps(input_value)
                elif isinstance(input_value, str):
                    arguments = input_value
                else:
                    arguments = ""
                if name and not arguments:
                    arguments = "{}"

                tool_call = ChatCompletionDeltaToolCall(
                    id=delta.get("tool_use_id") or delta.get("id"),
                    type="function",
                    function=Function(
                        name=name,
                        arguments=arguments,
                    ),
                    index=tool_call_index if tool_call_index >= 0 else 0,
                )
                delta["tool_calls"] = [tool_call]
                delta.pop("type", None)
                delta.pop("tool_use_id", None)
                delta.pop("input", None)
                delta.pop("name", None)
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
        - consecutive assistant messages (text-only followed by tool_call-only,
          as the OpenAI Agents SDK replays them) -> merged into a single
          assistant message with both text and tool_use blocks. Snowflake
          Cortex rejects role-alternation violations with 390142.
        - OpenAI `annotations: []` on content blocks -> stripped. Snowflake
          Cortex rejects unknown fields with 390142.
        - list-form `content` ([{"type": "text", "text": ...}], the shape the
          Agents SDK replays a prior assistant turn in) -> flattened to a
          plain string. Snowflake Cortex rejects array-form `content` with
          390142, so a plain multi-turn text conversation otherwise fails on
          the second turn.
        - content=None -> content="" (Snowflake requires non-null content)
        """
        transformed_messages: List[Dict[str, Any]] = []
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
                tool_use_blocks: List[Dict[str, Any]] = []

                for tool_call in msg_dict.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        function_data = tool_call.get("function", {})
                        tc_id = tool_call.get("id")
                        tc_name = function_data.get("name")
                        arguments_str = function_data.get("arguments", "{}")

                        if tc_id and tc_name:
                            tool_call_map[tc_id] = tc_name

                        # Empty/missing arguments → parameterless call. The
                        # Cortex-via-Bedrock streaming path leaves
                        # `arguments=""` on the SDK side; coerce to a valid
                        # empty object before serialization.
                        if not arguments_str:
                            arguments: Any = {}
                        else:
                            try:
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )
                            except json.JSONDecodeError:
                                arguments = {}

                        tool_use_blocks.append({
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": tc_id,
                                "name": tc_name,
                                "input": arguments,
                            },
                        })

                # Merge with the immediately preceding assistant message if it
                # carried only text and no content_list / tool_calls. The
                # OpenAI Agents SDK replays a tool-using turn as two separate
                # assistant messages (text first, tool_call second); Snowflake
                # Cortex expects a single assistant message per turn whose
                # content_list contains both text and tool_use blocks.
                text_prelude_blocks: List[Dict[str, Any]] = []
                if (
                    transformed_messages
                    and transformed_messages[-1].get("role") == "assistant"
                    and not transformed_messages[-1].get("content_list")
                    and not transformed_messages[-1].get("tool_calls")
                ):
                    prev = transformed_messages.pop()
                    text_prelude_blocks = _content_to_text_blocks(
                        prev.get("content")
                    )

                content_list = text_prelude_blocks + tool_use_blocks

                transformed_messages.append({
                    "role": "assistant",
                    # Flatten any list-form content to a string — Snowflake
                    # rejects array-form `content` with 390142.
                    "content": _content_to_text_string(msg_dict.get("content")),
                    "content_list": content_list,
                })

            else:
                if isinstance(message, dict):
                    msg_to_append = message.copy()
                else:
                    msg_to_append = dict(message)

                _strip_openai_annotations(msg_to_append)

                # Snowflake Cortex requires `content` to be a plain string, not
                # an array of content blocks. The OpenAI Agents SDK replays a
                # prior assistant turn as list-form content
                # ([{"type": "text", "text": "..."}]); left unflattened it is
                # rejected with 390142 on the next turn. Flatten to a string.
                content_value = msg_to_append.get("content")
                if isinstance(content_value, list):
                    msg_to_append["content"] = _content_to_text_string(content_value)
                    content_value = msg_to_append["content"]

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
