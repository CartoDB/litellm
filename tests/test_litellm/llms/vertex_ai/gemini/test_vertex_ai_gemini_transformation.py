from litellm.llms.vertex_ai.gemini.transformation import (
    check_if_part_exists_in_parts,
    _transform_request_body,
)


def test_check_if_part_exists_in_parts():
    parts = [
        {"text": "Hello", "thought": True},
        {"text": "World", "thought": False},
    ]
    part = {"text": "Hello", "thought": True}
    new_part = {"text": "Hello World", "thought": True}
    assert check_if_part_exists_in_parts(parts, part)
    assert not check_if_part_exists_in_parts(parts, new_part, ["thought"])
    assert check_if_part_exists_in_parts(parts, new_part, ["text"])


def test_check_if_part_exists_in_parts_camel_case_snake_case():
    """Test that function handles both camelCase and snake_case key variations"""
    # Test snake_case to camelCase matching
    parts_with_snake_case = [
        {
            "function_call": {
                "name": "get_current_weather",
                "args": {"location": "San Francisco, CA"},
            }
        },
        {"text": "Some other content"},
    ]

    part_with_camel_case = {
        "functionCall": {
            "name": "get_current_weather",
            "args": {"location": "San Francisco, CA"},
        }
    }

    # Should find match between function_call and functionCall
    assert check_if_part_exists_in_parts(parts_with_snake_case, part_with_camel_case)

    # Test camelCase to snake_case matching
    parts_with_camel_case = [
        {"functionCall": {"name": "calculate_sum", "args": {"a": 1, "b": 2}}}
    ]

    part_with_snake_case = {
        "function_call": {"name": "calculate_sum", "args": {"a": 1, "b": 2}}
    }

    # Should find match between functionCall and function_call
    assert check_if_part_exists_in_parts(parts_with_camel_case, part_with_snake_case)

    # Test no match when values differ
    part_with_different_values = {
        "function_call": {"name": "different_function", "args": {"x": 5}}
    }

    assert not check_if_part_exists_in_parts(
        parts_with_snake_case, part_with_different_values
    )

    # Test multiple keys with mixed casing
    parts_mixed = [
        {
            "function_call": {"name": "test"},
            "thoughtSignature": "reasoning",
            "text": "content",
        }
    ]

    part_mixed_casing = {
        "functionCall": {"name": "test"},
        "thought_signature": "reasoning",
        "text": "content",
    }

    assert check_if_part_exists_in_parts(parts_mixed, part_mixed_casing)


# Tests for issue #14556: Labels field provider-aware filtering
def test_google_genai_excludes_labels():
    """Test that Google GenAI/AI Studio endpoints exclude labels when custom_llm_provider='gemini'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Google GenAI/AI Studio should NOT include labels
    assert "labels" not in result
    assert "contents" in result


def test_vertex_ai_includes_labels():
    """Test that Vertex AI endpoints include labels when custom_llm_provider='vertex_ai'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Vertex AI SHOULD include labels
    assert "labels" in result
    assert result["labels"] == {"project": "test", "team": "ai"}



def test_metadata_to_labels_vertex_only():
    """Test that metadata->labels conversion only happens for Vertex AI"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {
                "user": "john_doe",
                "project": "test-project"
            }
        }
    }

    # Google GenAI/AI Studio should not include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="gemini",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" not in result

    # Vertex AI should include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" in result
    assert result["labels"] == {"user": "john_doe", "project": "test-project"}


def test_metadata_none_no_crash():
    """Test that metadata=None doesn't cause TypeError (bug fix for issue reported by Ana)"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {"metadata": None}  # Key exists but value is None

    # Should not crash with TypeError
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Should successfully return result without labels
    assert "contents" in result
    assert "labels" not in result


def test_metadata_empty_dict():
    """Test that empty metadata dict is handled correctly"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {"metadata": {}}  # Empty dict

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Should not crash and should not add labels
    assert "contents" in result
    assert "labels" not in result


def test_metadata_filters_non_string_values():
    """Test that only string values are converted to labels"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {
                "user": "john_doe",         # string - should be included
                "project": "test-project",  # string - should be included
                "count": 42,                # int - should be filtered out
                "enabled": True,            # bool - should be filtered out
                "config": {"key": "value"}, # dict - should be filtered out
                "items": ["a", "b"],        # list - should be filtered out
            }
        }
    }

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Only string values should be in labels
    assert "labels" in result
    assert result["labels"] == {"user": "john_doe", "project": "test-project"}
    assert "count" not in result["labels"]
    assert "enabled" not in result["labels"]
    assert "config" not in result["labels"]
    assert "items" not in result["labels"]
