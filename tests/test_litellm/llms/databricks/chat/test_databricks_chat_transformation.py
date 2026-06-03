import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
    _sanitize_empty_content,
    _strip_openai_annotations,
)


def test_transform_choices():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking.",
                                "signature": "ErcBCkgIAhABGAIiQMadog2CAJc8YJdce2Cmqvk0MFB+gGt4OyaH4c3l9p9v+0TKhYcNGliFkxddhCVkYR8zz8oaO1f3cHaEmYXN5SISDGAaomDR7CaTrhZxURoMbOR7AfFuHcIdVXFSIjC9ZamSyhzMg3maOtq2QHLXr6Z7tv0dut2S0Icdqk4g7MOFTSnCc0jA7lvnJyjI0wMqHR05PoVXEDSQjAV6NcUFkzFzp34z0xVMaK/VatCT",
                            }
                        ],
                    },
                    {"type": "text", "text": "# 5 Question and Answer Pairs"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "# 5 Question and Answer Pairs"
    assert choices[0].message.reasoning_content == "i'm thinking."
    assert choices[0].message.thinking_blocks is not None
    assert choices[0].message.tool_calls is None


def test_transform_choices_without_signature():
    """
    Test that the transformation works correctly when the signature field is missing
    from the summary, which occurs with new Databricks Foundation Models like
    databricks-gpt-oss-20b and databricks-gpt-oss-120b.
    """
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking without signature.",
                                # Note: no signature field here
                            }
                        ],
                    },
                    {"type": "text", "text": "Response without signature"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    # This should not raise a KeyError for missing signature
    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "Response without signature"
    assert choices[0].message.reasoning_content == "i'm thinking without signature."
    assert choices[0].message.thinking_blocks is not None
    assert len(choices[0].message.thinking_blocks) == 1

    # Verify the thinking block was created successfully without signature
    thinking_block = choices[0].message.thinking_blocks[0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == "i'm thinking without signature."


def test_convert_anthropic_tool_to_databricks_tool_with_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "description": "test description",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"]["description"] == "test description"


def test_convert_anthropic_tool_to_databricks_tool_without_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"].get("description") is None


def test_transform_choices_with_citations():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Blue",
                        "citations": [
                            {
                                "type": "char_location",
                                "cited_text": "The sky is blue.",
                                "document_index": 0,
                                "document_title": "My Document",
                                "start_char_index": 0,
                                "end_char_index": 50,
                            }
                        ],
                    }
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.provider_specific_fields == {
        "citations": [
            [
                {
                    "type": "char_location",
                    "cited_text": "The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "start_char_index": 0,
                    "end_char_index": 50,
                    "supported_text": "Blue",
                }
            ]
        ]
    }


def test_chunk_parser_with_citation():
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "delta": {
                    "content": [
                        {
                            "type": "text",
                            "text": "",
                            "citations": [
                                {
                                    "type": "char_location",
                                    "cited_text": "The sky is blue.",
                                    "document_index": 0,
                                    "document_title": "My Document",
                                    "start_char_index": 0,
                                    "end_char_index": 50,
                                }
                            ],
                        }
                    ],
                },
                "index": 0,
                "finish_reason": None,
            }
        ],
    }

    parsed = iterator.chunk_parser(chunk)
    assert parsed.choices[0].delta.provider_specific_fields == {
        "citation": {
            "type": "char_location",
            "cited_text": "The sky is blue.",
            "document_index": 0,
            "document_title": "My Document",
            "start_char_index": 0,
            "end_char_index": 50,
        }
    }


def test_chunk_parser_preserves_empty_object_tool_arguments():
    # Regression: a previous "avoid invalid json" guard rewrote tool_call
    # arguments from "{}" to "" when a single streaming chunk carried the
    # complete empty-object payload. That made the persisted assistant
    # message invalid on the next turn — downstream providers (Databricks
    # included) reject `arguments: ""` with a JSON parse error.
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_map_coordinates",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "index": 0,
                "finish_reason": None,
            }
        ],
    }

    parsed = iterator.chunk_parser(chunk)
    assert parsed.choices[0].delta.tool_calls[0].function.arguments == "{}"


def test_sanitize_empty_content_pops_none():
    message = {"role": "user", "content": None}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_pops_empty_string():
    message = {"role": "user", "content": ""}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_pops_single_empty_text_block():
    message = {"role": "user", "content": [{"type": "text", "text": ""}]}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_filters_empty_blocks_keeps_non_empty():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "  "},
        ],
    }
    _sanitize_empty_content(message)
    assert message["content"] == [{"type": "text", "text": "Hello"}]


def test_transform_messages_sanitizes_empty_content():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": "Hi"},
    ]
    result = config._transform_messages(
        messages=messages, model="databricks-claude", is_async=False
    )
    assert "content" not in result[0]
    assert result[1]["content"] == "Hi"


def test_strip_openai_annotations_removes_annotations_field():
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {"url": "https://example.com"},
                    }
                ],
            }
        ],
    }
    _strip_openai_annotations(message)
    assert message["content"] == [{"type": "text", "text": "Hello"}]


def test_strip_openai_annotations_leaves_string_content_untouched():
    message = {"role": "user", "content": "Hi"}
    _strip_openai_annotations(message)
    assert message["content"] == "Hi"


def test_strip_openai_annotations_noop_when_no_annotations():
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
    }
    _strip_openai_annotations(message)
    assert message["content"] == [{"type": "text", "text": "Hello"}]


def test_transform_messages_strips_annotations_from_assistant_content():
    # Regression: Databricks Model Serving rejects assistant messages whose
    # content blocks carry an `annotations` field (OpenAI chat-completion
    # citation parity). The Agents SDK persists this verbatim and replays it
    # on the next turn, causing a 400 BAD_REQUEST:
    #   messages.<n>.content.<m>.text.annotations: Extra inputs are not permitted
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello there",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {"url": "https://example.com"},
                        }
                    ],
                }
            ],
        },
    ]
    result = config._transform_messages(
        messages=messages, model="databricks-claude", is_async=False
    )
    assert result[1]["content"] == [{"type": "text", "text": "Hello there"}]
