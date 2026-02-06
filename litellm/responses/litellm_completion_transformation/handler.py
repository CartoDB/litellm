"""
Handler for transforming responses api requests to litellm.completion requests
"""

import uuid
from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse


class LiteLLMCompletionTransformationHandler:

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        litellm_completion_request: dict = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **litellm_completion_request,
            **kwargs,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                litellm_completion_request=litellm_completion_request,
            )

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:

        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(
            **acompletion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            # CARTO PATCH: Store session immediately in Redis to avoid batch processing delay
            # Store BOTH input messages AND the assistant response to preserve tool_calls with thought_signature
            if responses_api_response.id:
                session_id = kwargs.get("litellm_trace_id") or str(uuid.uuid4())
                current_messages = list(litellm_completion_request.get("messages", []))

                # Extract assistant message from the completion response
                assistant_message = _extract_assistant_message_from_response(litellm_completion_response)
                if assistant_message:
                    current_messages.append(assistant_message)

                await LiteLLMCompletionResponsesConfig._patch_store_session_in_redis(
                    response_id=responses_api_response.id,
                    session_id=session_id,
                    messages=current_messages
                )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get(
                    "custom_llm_provider"
                ),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                litellm_completion_request=litellm_completion_request,
            )


def _extract_assistant_message_from_response(response: ModelResponse) -> Optional[Dict]:
    """
    Extract the assistant message from a ModelResponse, preserving tool_calls with provider_specific_fields.

    This is critical for Gemini thinking models where thought_signature must be preserved
    for multi-turn conversations with tool calling.
    """
    if not response.choices:
        return None

    choice = response.choices[0]
    if not hasattr(choice, 'message') or not choice.message:
        return None

    msg = choice.message

    assistant_message: Dict[str, Any] = {
        "role": "assistant",
    }

    # Add content if present
    content = getattr(msg, 'content', None)
    if content is not None:
        assistant_message["content"] = content

    # Add tool_calls with provider_specific_fields (contains thought_signature for Gemini)
    tool_calls = getattr(msg, 'tool_calls', None)
    if tool_calls:
        serialized_tool_calls = []
        for tc in tool_calls:
            tool_call_dict: Dict[str, Any] = {
                "id": getattr(tc, 'id', None) or tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None),
                "type": getattr(tc, 'type', 'function') or tc.get('type', 'function') if isinstance(tc, dict) else getattr(tc, 'type', 'function'),
            }

            # Extract function details
            fn = tc.get('function') if isinstance(tc, dict) else getattr(tc, 'function', None)
            if fn:
                fn_dict: Dict[str, Any] = {
                    "name": fn.get('name') if isinstance(fn, dict) else getattr(fn, 'name', ''),
                    "arguments": fn.get('arguments', '') if isinstance(fn, dict) else getattr(fn, 'arguments', ''),
                }
                # Preserve function-level provider_specific_fields (contains thought_signature)
                fn_provider_fields = fn.get('provider_specific_fields') if isinstance(fn, dict) else getattr(fn, 'provider_specific_fields', None)
                if fn_provider_fields:
                    fn_dict["provider_specific_fields"] = fn_provider_fields
                tool_call_dict["function"] = fn_dict

            # Also check for tool_call-level provider_specific_fields
            tc_provider_fields = tc.get('provider_specific_fields') if isinstance(tc, dict) else getattr(tc, 'provider_specific_fields', None)
            if tc_provider_fields:
                tool_call_dict["provider_specific_fields"] = tc_provider_fields

            serialized_tool_calls.append(tool_call_dict)

        assistant_message["tool_calls"] = serialized_tool_calls

    # Preserve message-level provider_specific_fields
    msg_provider_fields = getattr(msg, 'provider_specific_fields', None)
    if msg_provider_fields:
        assistant_message["provider_specific_fields"] = msg_provider_fields

    # Preserve thinking_blocks if present (Anthropic/Claude)
    thinking_blocks = getattr(msg, 'thinking_blocks', None)
    if thinking_blocks:
        assistant_message["thinking_blocks"] = thinking_blocks

    # Preserve reasoning_content if present
    reasoning_content = getattr(msg, 'reasoning_content', None)
    if reasoning_content:
        assistant_message["reasoning_content"] = reasoning_content

    return assistant_message
