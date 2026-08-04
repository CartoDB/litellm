"""
Translates from OpenAI's `/v1/chat/completions` to Databricks' `/chat/completions`
"""

import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import httpx
from pydantic import BaseModel

from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _handle_invalid_parallel_tool_calls,
    _should_convert_tool_call_to_json_mode,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    strip_name_from_message,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.anthropic import AllAnthropicToolsValues
from litellm.types.llms.databricks import (
    AllDatabricksContentValues,
    DatabricksChoice,
    DatabricksFunction,
    DatabricksResponse,
    DatabricksTool,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Message,
    ModelResponse,
    ModelResponseStream,
    ProviderField,
    Usage,
)

from ...anthropic.chat.transformation import (
    REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT,
    AnthropicConfig,
)
from ...openai_like.chat.transformation import OpenAILikeChatConfig
from ..common_utils import DatabricksBase, DatabricksException


def _sanitize_empty_content(message_dict: dict[str, Any]) -> None:
    """
    Remove or filter content so empty text blocks are not sent.
    Databricks Model Serving uses Anthropic Messages API spec and rejects empty text blocks.
    """
    content = message_dict.get("content")
    if content is None:
        message_dict.pop("content", None)
        return
    if isinstance(content, str):
        if not content.strip():
            message_dict.pop("content")
        return
    if isinstance(content, list):
        if not content:
            message_dict.pop("content")
            return
        filtered = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "text" and not (block.get("text") or "").strip())
        ]
        if not filtered:
            message_dict.pop("content")
        else:
            message_dict["content"] = filtered


def _split_parallel_tool_calls(messages: list[AllMessageValues]) -> list[AllMessageValues]:
    """
    Databricks (OpenAI-compatible serving) rejects a ``tool`` message unless the
    message immediately before it carries ``tool_calls``. A single assistant turn
    with parallel tool calls is followed by one ``tool`` message per call, so every
    result after the first is preceded by another ``tool`` message and 400s. Re-emit
    each result right after an assistant message holding only its matching call:
    ``assistant(tool_calls=[A, B]), tool(A), tool(B)`` becomes
    ``assistant(tool_calls=[A]), tool(A), assistant(tool_calls=[B]), tool(B)``.

    Left untouched (no-op) when the turn is already valid or the history is
    malformed, so no tool call is ever dropped.
    """

    def _expand(
        assistant: ChatCompletionAssistantMessage,
        calls_by_id: dict[Optional[str], ChatCompletionAssistantToolCall],
        tool_messages: list[ChatCompletionToolMessage],
    ) -> Iterator[AllMessageValues]:
        for position, tool_message in enumerate(tool_messages):
            matched_call = calls_by_id[tool_message["tool_call_id"]]
            if position == 0:
                yield cast(AllMessageValues, {**assistant, "tool_calls": [matched_call]})
            else:
                yield ChatCompletionAssistantMessage(role="assistant", tool_calls=[matched_call])
            yield tool_message

    def _generate() -> Iterator[AllMessageValues]:
        index = 0
        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls") if message["role"] == "assistant" else None
            if not tool_calls or len(tool_calls) < 2:
                yield message
                index += 1
                continue
            end = index + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            tool_messages = cast(list[ChatCompletionToolMessage], messages[index + 1 : end])
            calls_by_id = {call["id"]: call for call in tool_calls}
            result_ids = {tool_message["tool_call_id"] for tool_message in tool_messages}
            if len(tool_messages) == len(tool_calls) and set(calls_by_id) == result_ids:
                yield from _expand(cast(ChatCompletionAssistantMessage, message), calls_by_id, tool_messages)
                index = end
            else:
                yield message
                index += 1

    return list(_generate())


def _normalize_empty_tool_call_arguments(message_dict: dict[str, Any]) -> None:
    """
    Restore `tool_calls[].function.arguments` to a valid JSON empty object
    when it is missing or an empty string.

    Databricks's streaming protocol emits parameterless tool_call arguments
    as one or two empty-string deltas (no `{}`), which accumulate to `""` in
    the consumer. When that assistant message is replayed on a follow-up
    turn (e.g. via the OpenAI Agents SDK conversation history), Databricks
    Model Serving rejects it with:

        INVALID_PARAMETER_VALUE: Param 'arguments' in the tool_calls
        function specification is not a valid JSON string. No content to
        map due to end-of-input

    Coercing empty/missing arguments to `"{}"` is safe because that is the
    canonical JSON representation of a parameterless call.
    """
    tool_calls = message_dict.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        if not fn.get("arguments"):
            fn["arguments"] = "{}"


def _strip_openai_annotations(message_dict: dict[str, Any]) -> None:
    """
    Remove the OpenAI-only `annotations` field from each content block.
    Databricks Model Serving uses Anthropic Messages API spec and rejects
    `annotations` with `Extra inputs are not permitted`. The field appears
    on chat-completion assistant messages emitted by OpenAI-compatible
    providers (for citation parity with the Responses API) and survives the
    Agents SDK replay on every follow-up turn.
    """
    content = message_dict.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and "annotations" in block:
            block.pop("annotations", None)


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class DatabricksConfig(DatabricksBase, OpenAILikeChatConfig, AnthropicConfig):
    """
    Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    n: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop: Optional[Union[List[str], str]] = None,
        n: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "databricks"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Databricks API Key.",
                field_value="dapi...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Databricks API Base.",
                field_value="https://adb-..",
            ),
        ]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        custom_user_agent = (
            optional_params.pop("user_agent", None)
            or optional_params.pop("databricks_user_agent", None)
            or litellm_params.get("user_agent")
            or os.getenv("LITELLM_USER_AGENT")
            or os.getenv("DATABRICKS_USER_AGENT")
        )

        api_base, headers = self.databricks_validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=headers,
            custom_user_agent=custom_user_agent,
        )
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = self._get_api_base(api_base)
        complete_url = f"{api_base}/chat/completions"
        return complete_url

    def get_supported_openai_params(self, model: Optional[str] = None) -> list:
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "response_format",
            "tools",
            "tool_choice",
            "reasoning_effort",
            "thinking",
        ]

    def convert_anthropic_tool_to_databricks_tool(
        self, tool: Optional[AllAnthropicToolsValues]
    ) -> Optional[DatabricksTool]:
        if tool is None:
            return None

        function_params: DatabricksFunction = {
            "name": tool["name"],
            "parameters": cast(dict, tool.get("input_schema") or {}),
        }

        description = tool.get("description")
        if description is not None:
            function_params["description"] = cast(Union[dict, str], description)

        return DatabricksTool(
            type="function",
            function=function_params,
        )

    def _map_openai_to_dbrx_tool(self, model: str, tools: List) -> List[DatabricksTool]:
        if "claude" not in model:
            return tools

        anthropic_tools, _ = self._map_tools(tools=tools)
        databricks_tools = [
            cast(DatabricksTool, self.convert_anthropic_tool_to_databricks_tool(tool)) for tool in anthropic_tools
        ]
        return databricks_tools

    def map_response_format_to_databricks_tool(
        self,
        model: str,
        value: Optional[dict],
        optional_params: dict,
        is_thinking_enabled: bool,
    ) -> Optional[DatabricksTool]:
        if value is None:
            return None

        tool = self.map_response_format_to_anthropic_tool(value, optional_params, is_thinking_enabled)

        databricks_tool = self.convert_anthropic_tool_to_databricks_tool(tool)
        return databricks_tool

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List["ChatCompletionToolParam"]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List["ChatCompletionToolParam"]]]:
        """
        Override the parent class method to preserve cache_control for models on Databricks.
        Databricks supports Anthropic-style cache control for Claude models.
        Databricks ignores the cache_control flag with other models.
        """
        return messages, tools

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        is_thinking_enabled = self.is_thinking_enabled(non_default_params)
        mapped_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        if "tools" in mapped_params:
            mapped_params["tools"] = self._map_openai_to_dbrx_tool(model=model, tools=mapped_params["tools"])
        if "max_completion_tokens" in non_default_params and replace_max_completion_tokens_with_max_tokens:
            mapped_params["max_tokens"] = non_default_params["max_completion_tokens"]
            mapped_params.pop("max_completion_tokens", None)

        if "response_format" in non_default_params and "claude" in model:
            _tool = self.map_response_format_to_databricks_tool(
                model,
                non_default_params["response_format"],
                mapped_params,
                is_thinking_enabled,
            )

            if _tool is not None:
                self._add_tools_to_optional_params(optional_params=optional_params, tools=[_tool])
                optional_params["json_mode"] = True
                if not is_thinking_enabled:
                    _tool_choice = ChatCompletionToolChoiceObjectParam(
                        type="function",
                        function=ChatCompletionToolChoiceFunctionParam(name=RESPONSE_FORMAT_TOOL_NAME),
                    )
                    optional_params["tool_choice"] = _tool_choice
            optional_params.pop("response_format", None)

        if "reasoning_effort" in non_default_params and "claude" in model:
            reasoning_effort_value = non_default_params.get("reasoning_effort")
            mapped_thinking = AnthropicConfig._map_reasoning_effort(
                reasoning_effort=reasoning_effort_value,
                model=model,
                custom_llm_provider="databricks",
                llm_provider="databricks",
            )
            if mapped_thinking is None:
                optional_params.pop("thinking", None)
                optional_params.pop("output_config", None)
            else:
                optional_params["thinking"] = mapped_thinking
                if AnthropicConfig._is_adaptive_thinking_model(model, "databricks"):
                    mapped_effort: Optional[str] = None
                    if isinstance(reasoning_effort_value, str):
                        mapped_effort = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(reasoning_effort_value)
                    if mapped_effort is None:
                        AnthropicConfig._raise_invalid_reasoning_effort(
                            model=model,
                            value=reasoning_effort_value,
                            llm_provider="databricks",
                        )
                    optional_params["output_config"] = {"effort": mapped_effort}
            optional_params.pop("reasoning_effort", None)
        self.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params, optional_params=mapped_params
        )

        return mapped_params

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Databricks doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Databricks does not support:
        - 'name' in user message.
        """
        new_messages = []
        for idx, message in enumerate(messages):
            if isinstance(message, BaseModel):
                _message = message.model_dump(exclude_none=True)
            else:
                _message = message
            _message = strip_name_from_message(_message, allowed_name_roles=["user"])
            if "cache_control" in _message and isinstance(_message.get("content"), str):
                _message = self._move_cache_control_into_string_content_block(_message)
            _sanitize_empty_content(cast(dict[str, Any], _message))
            _strip_openai_annotations(cast(dict[str, Any], _message))
            _normalize_empty_tool_call_arguments(cast(dict[str, Any], _message))
            new_messages.append(_message)

        if "claude" not in model:
            new_messages = _split_parallel_tool_calls(cast(list[AllMessageValues], new_messages))

        if is_async:
            return super()._transform_messages(messages=new_messages, model=model, is_async=cast(Literal[True], True))
        else:
            return super()._transform_messages(messages=new_messages, model=model, is_async=cast(Literal[False], False))

    def _move_cache_control_into_string_content_block(self, message: AllMessageValues) -> AllMessageValues:
        """
        Moves message-level cache_control into a content block when content is a string.

        Transforms:
            {"role": "user", "content": "text", "cache_control": {...}}
        Into:
            {"role": "user", "content": [{"type": "text", "text": "text", "cache_control": {...}}]}

        This is required for Anthropic's prompt caching API when cache_control is specified
        at the message level but content is a simple string (not already an array of content blocks).
        """
        content = message.get("content")
        transformed_message = cast(dict[str, Any], message.copy())
        cache_control = transformed_message.pop("cache_control")
        transformed_message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": cache_control,
            }
        ]
        return cast(AllMessageValues, transformed_message)

    @staticmethod
    def extract_content_str(
        content: Optional[AllDatabricksContentValues],
    ) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            content_str = ""
            for item in content:
                if item.get("type") == "text":
                    text_value = item.get("text", "")
                    content_str += str(text_value) if text_value is not None else ""
            return content_str
        else:
            raise Exception(f"Unsupported content type: {type(content)}")

    @staticmethod
    def extract_reasoning_content(
        content: Optional[AllDatabricksContentValues],
    ) -> Tuple[
        Optional[str],
        Optional[List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]],
    ]:
        """
        Extract and return the reasoning content and thinking blocks
        """
        if content is None:
            return None, None
        thinking_blocks: Optional[List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]] = None
        reasoning_content: Optional[str] = None
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "reasoning":
                    summary_list = item.get("summary", [])
                    if isinstance(summary_list, list):
                        for sum in summary_list:
                            if reasoning_content is None:
                                reasoning_content = ""
                            reasoning_content += sum["text"]
                            thinking_block = ChatCompletionThinkingBlock(
                                type="thinking",
                                thinking=sum.get("text", ""),
                                signature=sum.get("signature", ""),
                            )
                            if thinking_blocks is None:
                                thinking_blocks = []
                            thinking_blocks.append(thinking_block)
        return reasoning_content, thinking_blocks

    @staticmethod
    def extract_citations(
        content: Optional[AllDatabricksContentValues],
    ) -> Optional[List[Any]]:
        if content is None:
            return None
        citations = []
        if isinstance(content, list):
            for item in content:
                text = item.get("text", None)
                if citations_item := item.get("citations"):
                    citations.append([{**citation, "supported_text": text} for citation in citations_item])
        return citations or None

    def _transform_dbrx_choices(
        self, choices: List[DatabricksChoice], json_mode: Optional[bool] = None
    ) -> List[Choices]:
        transformed_choices = []

        for choice in choices:
            tool_calls = choice["message"].get("tool_calls", None)
            if tool_calls is not None:
                _openai_tool_calls = []
                for _tc in tool_calls:
                    _openai_tc = ChatCompletionMessageToolCall(**_tc)  # type: ignore
                    _openai_tool_calls.append(_openai_tc)
                fixed_tool_calls = _handle_invalid_parallel_tool_calls(_openai_tool_calls)

                if fixed_tool_calls is not None:
                    tool_calls = fixed_tool_calls

            translated_message: Optional[Message] = None
            finish_reason: Optional[str] = None
            if tool_calls and _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls,
                convert_tool_call_to_json_mode=json_mode,
            ):
                json_mode_content_str: Optional[str] = str(tool_calls[0]["function"].get("arguments", "")) or None
                if json_mode_content_str is not None:
                    translated_message = Message(content=json_mode_content_str)
                    finish_reason = "stop"

            if translated_message is None:
                content_str = DatabricksConfig.extract_content_str(choice["message"]["content"])

                (
                    reasoning_content,
                    thinking_blocks,
                ) = DatabricksConfig.extract_reasoning_content(choice["message"].get("content"))

                citations = DatabricksConfig.extract_citations(choice["message"].get("content"))

                translated_message = Message(
                    role="assistant",
                    content=content_str,
                    reasoning_content=reasoning_content,
                    thinking_blocks=thinking_blocks,
                    tool_calls=choice["message"].get("tool_calls"),
                    provider_specific_fields=({"citations": citations} if citations is not None else None),
                )

            if finish_reason is None:
                finish_reason = choice["finish_reason"]

            translated_choice = Choices(
                finish_reason=finish_reason,
                index=choice["index"],
                message=translated_message,
                logprobs=None,
                enhancements=None,
            )

            transformed_choices.append(translated_choice)

        return transformed_choices

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
        redacted_request_data = self.redact_sensitive_data(request_data)

        logging_obj.post_call(
            input=messages,
            api_key="[REDACTED]",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": redacted_request_data},
        )

        try:
            completion_response = DatabricksResponse(**raw_response.json())  # type: ignore
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise DatabricksException(
                message="Unable to get json response - {}, Original Response: {}".format(str(e), raw_response.text),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        _custom_llm_provider = litellm_params.get("custom_llm_provider") or "databricks"
        _response_model = completion_response.get("model") or ""
        model_response.model = f"{_custom_llm_provider}/{_response_model}"
        model_response.id = completion_response["id"]
        model_response.created = completion_response["created"]
        setattr(model_response, "usage", Usage(**completion_response["usage"]))

        model_response.choices = self._transform_dbrx_choices(  # type: ignore
            choices=completion_response["choices"],
            json_mode=json_mode,
        )

        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return DatabricksChatResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class DatabricksChatResponseIterator(BaseModelResponseIterator):
    def __init__(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(streaming_response, sync_stream)

        self.json_mode = json_mode
        self._last_function_name = None

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            translated_choices = []
            for choice in chunk["choices"]:
                tool_calls = choice["delta"].get("tool_calls")
                if tool_calls and self.json_mode:
                    from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
                    from litellm.llms.base_llm.base_utils import (
                        _convert_tool_response_to_message,
                    )

                    function_name = tool_calls[0].get("function", {}).get("name")
                    if function_name is not None:
                        self._last_function_name = function_name

                    if (
                        self._last_function_name == RESPONSE_FORMAT_TOOL_NAME
                        or function_name == RESPONSE_FORMAT_TOOL_NAME
                    ):
                        message = _convert_tool_response_to_message(tool_calls)
                        if message is not None:
                            if message.content == "{}":
                                message.content = ""
                            choice["delta"]["content"] = message.content
                            choice["delta"]["tool_calls"] = None
                elif tool_calls:
                    for _tc in tool_calls:
                        fn = _tc.get("function") or {}
                        if fn.get("name") and not fn.get("arguments"):
                            fn["arguments"] = "{}"
                            _tc["function"] = fn
                if isinstance(choice["delta"].get("content"), list) and (
                    content := choice["delta"]["content"]
                ):
                    if citations := content[0].get("citations"):
                        choice["delta"].setdefault("provider_specific_fields", {})["citation"] = citations[0]
                content_str = DatabricksConfig.extract_content_str(choice["delta"].get("content"))

                (
                    reasoning_content,
                    thinking_blocks,
                ) = DatabricksConfig.extract_reasoning_content(choice["delta"].get("content"))

                choice["delta"]["content"] = content_str
                choice["delta"]["reasoning_content"] = reasoning_content
                choice["delta"]["thinking_blocks"] = thinking_blocks
                translated_choices.append(choice)
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                model=chunk["model"],
                choices=translated_choices,
            )
        except KeyError as e:
            raise DatabricksException(
                message=f"KeyError: {e}, Got unexpected response from Databricks: {chunk}",
                status_code=400,
            )
        except Exception as e:
            raise e
