#!/usr/bin/env python3
"""Preflight checks for LLM-Sana GRPO runs."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _status(ok: bool, name: str, detail: str) -> bool:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")
    return ok


def _check_path(path: str | None, name: str, required: bool = True) -> bool:
    if not path:
        return _status(not required, name, "not set")
    p = Path(path).expanduser()
    return _status(p.exists(), name, str(p))


def _check_model_ref(value: str | None, name: str, local_files_only: bool) -> bool:
    if not value:
        return _status(False, name, "not set")
    path = Path(value).expanduser()
    if path.exists():
        return _status(True, name, f"local: {path}")
    is_hf_id = "/" in value and not value.startswith(("/", "./", "../"))
    if is_hf_id and not local_files_only:
        return _status(True, name, f"Hugging Face ID: {value}")
    return _status(False, name, f"missing local path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sana-model", required=True)
    parser.add_argument("--reward-mode", default="biovil_label_raddino")
    parser.add_argument("--biovil-path", default=None)
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--raddino-path", default=None)
    parser.add_argument("--raddino-cache", default=None)
    parser.add_argument("--rollout-backend", default="vllm", choices=["vllm", "sglang", "trtllm"])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    for module in ("torch", "transformers", "verl", "ray", "diffusers", args.rollout_backend):
        spec = importlib.util.find_spec(module)
        if not _status(spec is not None, module, str(spec)):
            failures.append(f"{module} is not importable")

    try:
        import torch

        _status(True, "python", sys.executable)
        _status(True, "torch cuda", f"available={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch import failed: {exc}")

    if not _check_model_ref(args.model, "llm model", args.local_files_only):
        failures.append("LLM model path is missing")
    if not _check_model_ref(args.sana_model, "sana model", args.local_files_only):
        failures.append("Sana model path is missing")

    mode_parts = set(args.reward_mode.split("_"))
    if "biovil" in mode_parts:
        if not _check_model_ref(args.biovil_path, "BioViL-T model", args.local_files_only):
            failures.append("BioViL-T path is required by reward mode")
        if not _status(importlib.util.find_spec("health_multimodal") is not None, "health_multimodal", "BioViL-T dependency"):
            failures.append("health_multimodal is required by BioViL-T reward")
    if "label" in mode_parts:
        if not _check_path(args.classifier_checkpoint, "label classifier checkpoint"):
            failures.append("classifier checkpoint is required by reward mode")
        if not _status(importlib.util.find_spec("timm") is not None, "timm", "label classifier dependency"):
            failures.append("timm is required by label consistency reward")
    if "raddino" in mode_parts:
        if not _check_model_ref(args.raddino_path, "RadDINO model", args.local_files_only):
            failures.append("RadDINO path is required by reward mode")
        if not _check_path(args.raddino_cache, "RadDINO reference cache"):
            failures.append("RadDINO cache is required by reward mode; build it before v3 training")

    if failures:
        print("\nPreflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 2

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
