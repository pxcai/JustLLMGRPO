"""Rewrite source CXR prompts with a trained JustLLMGRPO policy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_sana.prompts import build_messages, extract_optimized_prompt


def render_chat(tokenizer, source_prompt: str) -> str:
    messages = build_messages(source_prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"


def extract_or_fallback(response: str, source_prompt: str) -> str:
    optimized = extract_optimized_prompt(response)
    return optimized or str(source_prompt).strip()


def _load_transformers(model_ref: str, device: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def rewrite_with_transformers(prompts: list[str], args: argparse.Namespace) -> dict[str, str]:
    tokenizer, model = _load_transformers(args.model, args.device, args.local_files_only)
    rewritten: dict[str, str] = {}
    do_sample = args.temperature > 0

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="LLM rewriting"):
        batch = prompts[start : start + args.batch_size]
        rendered = [render_chat(tokenizer, prompt) for prompt in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_ids, skip_special_tokens=False)
        for source, response in zip(batch, responses, strict=True):
            rewritten[source] = extract_or_fallback(response, source)
    return rewritten


def rewrite_with_vllm(prompts: list[str], args: argparse.Namespace) -> dict[str, str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        download_dir=args.download_dir,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    rewritten: dict[str, str] = {}
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="LLM rewriting"):
        batch = prompts[start : start + args.batch_size]
        rendered = [render_chat(tokenizer, prompt) for prompt in batch]
        outputs = llm.generate(rendered, sampling, use_tqdm=False)
        for source, output in zip(batch, outputs, strict=True):
            response = output.outputs[0].text if output.outputs else ""
            rewritten[source] = extract_or_fallback(response, source)
    return rewritten


def rewrite_csv(args: argparse.Namespace) -> Path:
    frame = pd.read_csv(args.input_csv)
    if args.prompt_column not in frame.columns:
        raise ValueError(f"Missing prompt column {args.prompt_column!r} in {args.input_csv}")

    source = frame[args.prompt_column].fillna("").astype(str).str.strip()
    frame = frame.loc[source != ""].copy()
    frame[args.source_column] = source[source != ""].values
    unique_prompts = frame[args.source_column].drop_duplicates().tolist()

    if args.backend == "vllm":
        prompt_map = rewrite_with_vllm(unique_prompts, args)
    elif args.backend == "transformers":
        prompt_map = rewrite_with_transformers(unique_prompts, args)
    else:
        try:
            prompt_map = rewrite_with_vllm(unique_prompts, args)
        except Exception as exc:  # noqa: BLE001
            print(f"vLLM failed ({exc}); falling back to Transformers.", file=sys.stderr)
            prompt_map = rewrite_with_transformers(unique_prompts, args)

    frame[args.output_column] = frame[args.source_column].map(prompt_map).fillna(frame[args.source_column])
    if args.replace_prompt_column:
        frame[args.prompt_column] = frame[args.output_column]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} rows and {len(unique_prompts)} unique rewrites to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", required=True, help="Merged JustLLMGRPO checkpoint or Hugging Face model ID.")
    parser.add_argument("--prompt-column", default="annotated_prompt")
    parser.add_argument("--source-column", default="source_prompt")
    parser.add_argument("--output-column", default="optimized_prompt")
    parser.add_argument("--replace-prompt-column", action="store_true")
    parser.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rewrite_csv(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
