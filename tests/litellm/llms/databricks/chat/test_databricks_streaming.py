"""
Unit tests for Databricks chat streaming, specifically for GPT-5 models.

Tests cover:
- Empty delta handling in finish chunks
- Content extraction from various delta formats
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.databricks.chat.transformation import DatabricksChatResponseIterator


class TestDatabricksStreamingEmptyDelta:
    """Tests for handling empty delta in Databricks streaming responses."""

    def _create_iterator(self):
        """Create a minimal DatabricksChatResponseIterator for testing."""
        return DatabricksChatResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

    def test_chunk_parser_handles_empty_delta_finish_chunk(self):
        """
        Test that GPT-5 finish chunks with empty delta {} are handled correctly.

        GPT-5 sends: {'delta': {}, 'finish_reason': 'stop'}
        This should not raise KeyError: 'content'
        """
        iterator = self._create_iterator()

        # Simulate GPT-5 finish chunk with empty delta
        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "gpt-5-2025-08-07",
            "choices": [
                {
                    "index": 0,
                    "delta": {},  # Empty delta - the problematic case
                    "finish_reason": "stop"
                }
            ]
        }

        # Should not raise KeyError
        result = iterator.chunk_parser(chunk)

        assert result is not None
        assert result.choices[0].finish_reason == "stop"

    def test_chunk_parser_handles_delta_with_content_string(self):
        """
        Test that normal content string in delta is handled correctly.
        """
        iterator = self._create_iterator()

        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "gpt-5-2025-08-07",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }
            ]
        }

        result = iterator.chunk_parser(chunk)

        assert result is not None
        assert result.choices[0].delta.content == "Hello"

    def test_chunk_parser_handles_delta_with_none_content(self):
        """
        Test that delta with explicit None content is handled correctly.
        """
        iterator = self._create_iterator()

        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "gpt-5-2025-08-07",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None},
                    "finish_reason": None
                }
            ]
        }

        result = iterator.chunk_parser(chunk)

        assert result is not None

    def test_chunk_parser_handles_delta_with_role_only(self):
        """
        Test that delta with only role (no content) is handled correctly.
        This is common for the first chunk in a stream.
        """
        iterator = self._create_iterator()

        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "gpt-5-2025-08-07",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }

        result = iterator.chunk_parser(chunk)

        assert result is not None

    def test_chunk_parser_handles_delta_with_list_content(self):
        """
        Test that delta with list content format is handled correctly.
        Some models return content as a list of content items.
        """
        iterator = self._create_iterator()

        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "databricks-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [
                            {"type": "text", "text": "Hello world"}
                        ]
                    },
                    "finish_reason": None
                }
            ]
        }

        result = iterator.chunk_parser(chunk)

        assert result is not None
        # Content should be extracted as string
        assert result.choices[0].delta.content == "Hello world"

    def test_chunk_parser_handles_delta_with_empty_list_content(self):
        """
        Test that delta with empty list content is handled correctly.
        """
        iterator = self._create_iterator()

        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1768813003,
            "model": "databricks-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": []},
                    "finish_reason": None
                }
            ]
        }

        result = iterator.chunk_parser(chunk)

        assert result is not None
