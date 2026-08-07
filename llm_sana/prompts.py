"""Prompt templates for LLM-planned chest X-ray generation."""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert prompt optimizer for chest X-ray diffusion models."
)

USER_TEMPLATE = """Source diffusion prompt:
{original_prompt}

Rewrite the source prompt into a chest X-ray prompt that is better suited for diffusion generation.

Guidelines:
- Preserve the source meaning.
- Feel free to rewrite, compress, and reorder the wording to make it better suited for chest X-ray generation.
- Do not add new clinical facts that are not already present in the source.

You can reason in <think></think>.
After </think>, put the final result in exactly one <optimized_prompt></optimized_prompt> block.
Do not output any other text."""


_OPTIMIZED_PROMPT_RE = re.compile(
    r"<optimized_prompt>(.*?)</optimized_prompt>",
    flags=re.IGNORECASE | re.DOTALL,
)


def build_messages(original_prompt: str) -> list[dict[str, str]]:
    """Build Qwen-style chat messages for prompt rewriting."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(original_prompt=str(original_prompt).strip()),
        },
    ]


def extract_optimized_prompt(solution_str: str, max_chars: int = 1200) -> str:
    """Extract the rewritten diffusion prompt from a model response."""
    text = str(solution_str or "").strip()
    if not text:
        return ""

    marker = text.lower().rfind("</think>")
    if marker < 0:
        return ""

    text = text[marker + len("</think>") :].strip()
    tagged = _OPTIMIZED_PROMPT_RE.search(text)
    if tagged is None:
        return ""
    text = tagged.group(1).strip()

    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    lines = [line.strip(" \t\r\n-\"'") for line in text.splitlines() if line.strip()]
    if lines:
        text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip(" \"'")
    if max_chars > 0:
        text = text[:max_chars].strip()
    return text
