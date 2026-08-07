"""Run the complete trained-policy inference pipeline: LLM rewrite then Sana render."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--sana-model", default="raman07/CheXGenBench-Models-Sana-e20")
    parser.add_argument("--prompt-column", default="annotated_prompt")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--planner-backend", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--planner-batch-size", type=int, default=8)
    parser.add_argument("--sana-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rewritten_csv = output_dir / "optimized_prompts.csv"

    rewrite_command = [
        sys.executable,
        "-m",
        "inference.rewrite_prompts",
        "--input-csv",
        args.input_csv,
        "--output-csv",
        str(rewritten_csv),
        "--model",
        args.planner_model,
        "--prompt-column",
        args.prompt_column,
        "--backend",
        args.planner_backend,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--batch-size",
        str(args.planner_batch_size),
    ]
    generate_command = [
        sys.executable,
        "-m",
        "inference.generate_sana",
        "--input-csv",
        str(rewritten_csv),
        "--output-dir",
        str(output_dir),
        "--sana-model",
        args.sana_model,
        "--id-column",
        args.id_column,
        "--batch-size",
        str(args.sana_batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.local_files_only:
        rewrite_command.append("--local-files-only")
        generate_command.append("--local-files-only")
    if args.resume:
        generate_command.append("--resume")

    if not (args.resume and rewritten_csv.exists()):
        _run(rewrite_command)
    _run(generate_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
