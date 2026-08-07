"""Generate chest X-rays from optimized prompts with a frozen Sana model."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def stable_seed(base_seed: int, sample_id: str, source_prompt: str) -> int:
    material = f"{base_seed}:{sample_id}:{source_prompt}"
    return int(hashlib.sha1(material.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)


def safe_stem(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text or fallback)[:120]


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def generate_csv(args: argparse.Namespace) -> Path:
    from diffusers import SanaPipeline

    frame = pd.read_csv(args.input_csv)
    if args.prompt_column not in frame.columns:
        raise ValueError(f"Missing prompt column {args.prompt_column!r} in {args.input_csv}")
    if args.source_column not in frame.columns:
        frame[args.source_column] = frame[args.prompt_column]
    if args.id_column not in frame.columns:
        frame[args.id_column] = [str(index) for index in range(len(frame))]

    frame = frame.dropna(subset=[args.prompt_column]).copy()
    frame[args.prompt_column] = frame[args.prompt_column].astype(str).str.strip()
    frame = frame.loc[frame[args.prompt_column] != ""].reset_index(drop=True)
    if args.deduplicate:
        frame = frame.drop_duplicates(subset=[args.source_column]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(args.dtype)
    pipe = SanaPipeline.from_pretrained(
        args.sana_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    records = frame.to_dict("records")
    image_paths: list[str] = []
    seeds: list[int] = []
    generator_device = device if device.startswith("cuda") else "cpu"

    for start in tqdm(range(0, len(records), args.batch_size), desc="Sana generation"):
        batch = records[start : start + args.batch_size]
        prompts = [str(item[args.prompt_column]) for item in batch]
        batch_seeds = [
            stable_seed(args.seed, str(item[args.id_column]), str(item[args.source_column])) for item in batch
        ]
        paths = [
            image_dir
            / f"{start + offset:06d}_{safe_stem(item[args.id_column], f'sample_{start + offset:06d}')}.png"
            for offset, item in enumerate(batch)
        ]

        if args.resume and all(path.exists() for path in paths):
            image_paths.extend(str(path.relative_to(output_dir)) for path in paths)
            seeds.extend(batch_seeds)
            continue

        generators = [torch.Generator(device=generator_device).manual_seed(seed) for seed in batch_seeds]
        with torch.inference_mode():
            images = pipe(
                prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
            ).images
        for image, path in zip(images, paths, strict=True):
            image.convert("RGB").save(path)
            image_paths.append(str(path.relative_to(output_dir)))
        seeds.extend(batch_seeds)

    frame["generation_seed"] = seeds
    frame["generated_image"] = image_paths
    metadata = output_dir / "metadata.csv"
    frame.to_csv(metadata, index=False)
    print(f"Saved {len(frame)} generated CXRs to {image_dir}")
    print(f"Saved metadata to {metadata}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sana-model", default="raman07/CheXGenBench-Models-Sana-e20")
    parser.add_argument("--prompt-column", default="optimized_prompt")
    parser.add_argument("--source-column", default="source_prompt")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_csv(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
