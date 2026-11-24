from unittest.mock import patch

import pytest

from litellm.responses.bridge_check import responses_api_bridge_check


@pytest.mark.parametrize(
    "model",
    [
        "databricks-gpt-6-sol",
        "databricks-gpt-6-astra",
        "databricks/databricks-gpt-6-luna",
        "databricks-gpt-5-6-terra",
        "databricks-gpt-5-6-sol",
    ],
)
def test_responses_api_bridge_check_databricks_gpt_tools_plus_reasoning_routes_to_responses(model):
    with patch("litellm.responses.bridge_check.get_model_info_helper", return_value={"mode": "chat"}):
        model_info, _ = responses_api_bridge_check(
            model=model,
            custom_llm_provider="databricks",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="minimal",
        )
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize(
    "model, tools, reasoning_effort",
    [
        ("databricks-gpt-5-4-mini", [{"type": "function", "function": {"name": "f"}}], "minimal"),
        ("databricks-gpt-5", [{"type": "function", "function": {"name": "f"}}], "minimal"),
        ("databricks-gpt-6-sol", None, "minimal"),
        ("databricks-gpt-6-sol", [{"type": "function", "function": {"name": "f"}}], "none"),
        ("databricks-claude-opus-5-5", [{"type": "function", "function": {"name": "f"}}], "minimal"),
    ],
)
def test_responses_api_bridge_check_databricks_stays_on_chat(model, tools, reasoning_effort):
    with patch("litellm.responses.bridge_check.get_model_info_helper", return_value={"mode": "chat"}):
        model_info, _ = responses_api_bridge_check(
            model=model,
            custom_llm_provider="databricks",
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
    assert model_info.get("mode") != "responses"
