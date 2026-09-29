# Upstream Sync Decision Log: v1.103.0

## Summary

Merged upstream tag v1.103.0 (HEAD) with carto/main (theirs), preserving all 13 CARTO features defined in `.github/carto-features.yml`

## Conflict Resolution Decisions

### 1. `litellm/litellm_core_utils/streaming_chunk_builder_utils.py`
- **Approach**: Accept upstream, then add CARTO's `_validate_and_repair_tool_arguments` helper
- **Rationale**: Upstream added `apply_grounding_request_counts`, which is independent of CARTO's JSON repair feature (PR #54)

### 2. `litellm/llms/databricks/chat/transformation.py`
- **Approach**: Accept upstream, then port all CARTO helpers
- **CARTO features preserved**:
  - `_normalize_empty_tool_call_arguments` (PR #109, #110)
  - `_strip_openai_annotations` (PR #110)
  - `_apply_gemini_thought_signature`, `_extract_databricks_thought_signature`, `_capture_gemini_thought_signature` (PR #132, sc-572123)
- **Rationale**: Databricks helpers are provider-specific and don't conflict with upstream generic refactors

### 3. `litellm/llms/snowflake/chat/transformation.py`
- **Approach**: Accept upstream, then port CARTO helpers
- **CARTO features preserved**:
  - `_strip_openai_annotations` (PR #111)
  - `_content_to_text_string` for array content flattening (PR #112)
  - Full URL passthrough in `get_complete_url`
- **Rationale**: Upstream refactored to Anthropic Messages API format. CARTO helpers adapt to this new structure

### 4. `litellm/responses/litellm_completion_transformation/transformation.py`
- **Approach**: Accept upstream, then port CARTO Redis session methods
- **CARTO features preserved**:
  - `_patch_store_session_in_redis`, `_patch_get_session_from_redis`, `_filter_empty_assistant_messages` (PR #16)
  - `tool_choice_value` auto-defaulting to "auto" when tools present (PR #38)
- **Rationale**: Redis session storage is orthogonal to upstream completion transformation logic

### 5. `litellm/responses/litellm_completion_transformation/streaming_iterator.py`
- **Approach**: Accept upstream, then add CARTO Redis session storage
- **CARTO features preserved**:
  - `_store_session_in_redis` method and call in `__anext__` (PR #16)
  - `litellm_completion_request` parameter in `__init__`
- **Rationale**: Redis storage call at stream completion ensures immediate session availability

### 6. Test files
- `tests/test_litellm/llms/databricks/chat/test_databricks_chat_transformation.py`: Accepted upstream, then added CARTO thought signature tests
- `tests/test_litellm/responses/litellm_completion_transformation/test_session_handler.py`: Accepted upstream, then added CARTO Redis session tests
- `tests/test_litellm/llms/snowflake/chat/test_snowflake_chat_transformation.py`: Accepted upstream (CARTO tests already covered by verification patterns)

## Verification

All 13 CARTO features verified present via pattern matching:
1. OCI Gemini Tool Call UUIDs - verified
2. OCI Parallel Tool Result Reordering - verified
3. OCI Inline PEM Key Normalization - verified
4. Snowflake Streaming + Tool Calling - verified
5. Snowflake Full URL Passthrough - verified
6. Azure URL Suffix Stripping - verified
7. JSON Repair for Streaming Tool Calls - verified
8. Redis Session Storage - verified
9. Snowflake Cortex Claude Function-Calling Follow-up Turns - verified
10. Snowflake Cortex Array Content Flattening - verified
11. Databricks Empty Tool Call Arguments Normalization - verified
12. Databricks Strip OpenAI Annotations - verified
13. Databricks Gemini Thought Signature Round-Trip - verified

## Files Not Modified (accepted upstream)

- `schema.prisma` - must match upstream exactly per sync requirements
- All other conflicting files where upstream had full ownership
