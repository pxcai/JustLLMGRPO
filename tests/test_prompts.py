from llm_sana.prompts import build_messages, extract_optimized_prompt


def test_shared_prompt_contains_source_text():
    messages = build_messages("Small left pleural effusion.")
    assert messages[0]["role"] == "system"
    assert "Small left pleural effusion." in messages[1]["content"]


def test_extract_valid_optimized_prompt():
    response = "<think>shorten</think><optimized_prompt>Small left pleural effusion.</optimized_prompt>"
    assert extract_optimized_prompt(response) == "Small left pleural effusion."


def test_missing_format_is_invalid():
    assert extract_optimized_prompt("Small left pleural effusion.") == ""
