"""Online reward for LLM prompt planning followed by frozen Sana rendering.

This file is intended to be used as verl's custom reward function:

    reward.custom_reward_function.path=llm_sana/rewards/llm_sana_online_reward.py
    reward.custom_reward_function.name=compute_score

The policy response is text. We extract the optimized diffusion prompt, render one
image with a frozen Sana pipeline, then score that image with CheXGenBenchReward.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import torch

from llm_sana.prompts import extract_optimized_prompt
from llm_sana.rewards.chexgenbench_reward import CheXGenBenchReward, CheXGenBenchRewardConfig
from llm_sana.rewards.reward_subprocess import RewardSubprocessClient


_LLM_SANA_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SANA_MODEL = "raman07/CheXGenBench-Models-Sana-e20"
_DEFAULT_CLASSIFIER = _PROJECT_ROOT / "artifacts" / "best_classifier.pt"
_DEFAULT_BIOVIL = "microsoft/BiomedVLP-BioViL-T"
_DEFAULT_RADDINO = "microsoft/rad-dino"
_DEFAULT_RADDINO_CACHE = _PROJECT_ROOT / "artifacts" / "raddino_train20k_ref.npz"


class _OnlineRewardKey(NamedTuple):
    sana_model_path: str
    reward_mode: str
    device: str
    biovil_t_path: str
    classifier_checkpoint: str
    raddino_path: str
    raddino_reference_cache: str
    biovil_weight: float
    label_weight: float
    raddino_weight: float
    raddino_score_mode: str
    raddino_topk: int
    raddino_density_temperature: float
    raddino_cov_regularization: float
    raddino_global_density_weight: float
    raddino_local_topk_weight: float
    raddino_memorization_weight: float
    raddino_memorization_threshold: float
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    debug_image_dir: str


_ENGINE_CACHE: dict[_OnlineRewardKey, "LlmSanaOnlineReward"] = {}
_ENGINE_CACHE_LOCK = threading.Lock()
_SUBPROCESS_CLIENTS: dict[str, RewardSubprocessClient] = {}
_SUBPROCESS_CLIENTS_LOCK = threading.Lock()
_ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES: str | None = None
_RAY_REWARD_ACTORS_LOCK = threading.Lock()
_RAY_REWARD_SCHEDULER = None
_RAY_REWARD_WORKERS: dict[int, Any] = {}


class _RayRewardScheduler:
    def __init__(self, devices: list[str]):
        if not devices:
            raise ValueError("devices must be non-empty")
        self.devices = list(devices)
        self.inflight = [0 for _ in self.devices]

    def acquire(self) -> tuple[int, str, list[int]]:
        index = min(range(len(self.inflight)), key=lambda idx: self.inflight[idx])
        self.inflight[index] += 1
        return index, self.devices[index], list(self.inflight)

    def release(self, index: int) -> list[int]:
        if 0 <= index < len(self.inflight):
            self.inflight[index] = max(0, self.inflight[index] - 1)
        return list(self.inflight)


class _RayRewardGpuWorker:
    def __init__(self, cuda_visible_devices: str, batch_size: int = 1, max_wait_ms: int = 50):
        self.cuda_visible_devices = str(cuda_visible_devices)
        self.batch_size = max(1, int(batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.client = RewardSubprocessClient(self.cuda_visible_devices)
        self._condition = threading.Condition()
        self._queue: list[dict[str, Any]] = []
        self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._batch_thread.start()
        print(
            "[llm_sana_reward] started Ray reward GPU worker; "
            f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices!r}; "
            f"sana_batch_size={self.batch_size}; max_wait_ms={int(self.max_wait_s * 1000)}",
            flush=True,
        )

    def compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        item = {
            "payload": payload,
            "enqueued_at": time.perf_counter(),
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        with self._condition:
            self._queue.append(item)
            self._condition.notify()
        item["event"].wait()
        if item["error"] is not None:
            raise RuntimeError(item["error"])
        return item["result"]

    def _batch_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                deadline = time.monotonic() + self.max_wait_s
                while len(self._queue) < self.batch_size and self.max_wait_s > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                items = self._queue[: self.batch_size]
                del self._queue[: self.batch_size]

            payloads = [item["payload"] for item in items]
            batch_started_at = time.perf_counter()
            try:
                results = self.client.compute_batch(payloads)
                if len(results) != len(items):
                    raise RuntimeError(f"Batch result length mismatch: got {len(results)}, expected {len(items)}")
                if len(items) > 1:
                    print(
                        "[llm_sana_reward] processed Sana reward batch; "
                        f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices!r}; batch_size={len(items)}",
                        flush=True,
                    )
                for item, result in zip(items, results, strict=True):
                    result["timing_s/reward_queue_wait"] = batch_started_at - item["enqueued_at"]
                    result["timing_s/reward_worker_batch_size"] = float(len(items))
                    item["result"] = result
            except Exception as exc:  # noqa: BLE001
                for item in items:
                    item["error"] = str(exc)
            finally:
                for item in items:
                    item["event"].set()


def _env_or_default(name: str, default: Path | str | None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if default is None:
        return None
    return str(default)


def _split_cuda_visible_devices(cuda_visible_devices: str | None) -> list[str]:
    if not cuda_visible_devices:
        return []
    return [item.strip() for item in str(cuda_visible_devices).split(",") if item.strip()]


def _ray_reward_worker_index() -> int | None:
    try:
        import ray

        ctx = ray.get_runtime_context()
        actor_name = getattr(ctx, "get_actor_name", lambda: None)()
    except Exception:  # noqa: BLE001
        actor_name = None
    if not actor_name:
        return None
    match = re.search(r"reward_loop_worker_(\d+)", actor_name)
    if match is None:
        return None
    return int(match.group(1))


def _select_reward_cuda_visible_devices() -> str | None:
    global _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES
    if _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES is not None:
        return _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES

    raw_devices = os.environ.get("LLMSANA_REWARD_CUDA_VISIBLE_DEVICES")
    devices = _split_cuda_visible_devices(raw_devices)
    if len(devices) <= 1:
        _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES = raw_devices
        return _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES

    worker_index = _ray_reward_worker_index()
    if worker_index is None:
        worker_index = os.getpid()
    selected = devices[worker_index % len(devices)]
    _ASSIGNED_REWARD_CUDA_VISIBLE_DEVICES = selected
    os.environ["LLMSANA_REWARD_ASSIGNED_CUDA_VISIBLE_DEVICES"] = selected
    print(
        "[llm_sana_reward] assigned reward CUDA device; "
        f"pool={raw_devices!r}; selected={selected!r}; worker_index={worker_index}",
        flush=True,
    )
    return selected


def _safe_filename(text: str, suffix: str = ".png") -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text[:48]).strip("_")
    return f"{clean}_{digest}{suffix}" if clean else f"{digest}{suffix}"


def _relative_prompt_reward(delta: float) -> float:
    if delta > 0:
        return 1.0 + delta
    if delta >= -0.03:
        return 0.1
    return 0.0


def _normalize_ground_truth(ground_truth: Any, extra_info: dict[str, Any] | None) -> dict[str, Any]:
    extra_info = extra_info or {}
    if isinstance(ground_truth, dict):
        data = dict(ground_truth)
    else:
        data = {"original_prompt": str(ground_truth)}
    if not data.get("original_prompt"):
        data["original_prompt"] = extra_info.get("original_prompt", "")
    if "chexpert_labels" not in data:
        data["chexpert_labels"] = extra_info.get("chexpert_labels", {})
    if "row_id" not in data:
        data["row_id"] = extra_info.get("row_id", extra_info.get("index", ""))
    return data


class LlmSanaOnlineReward:
    """Render LLM-planned prompts with frozen Sana and score with CheXGenBench reward."""

    def __init__(
        self,
        *,
        sana_model_path: str,
        reward_config: CheXGenBenchRewardConfig,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = 42,
        debug_image_dir: str | None = None,
    ):
        self.sana_model_path = str(sana_model_path)
        self.reward = CheXGenBenchReward(reward_config)
        self.device = reward_config.device
        self.height = int(height)
        self.width = int(width)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.debug_image_dir = Path(debug_image_dir) if debug_image_dir else None
        self._pipe = None
        self._lock = threading.Lock()

    def _empty_result(self) -> dict[str, Any]:
        missing_score = float("nan")
        result: dict[str, Any] = {
            "score": 0.0,
            "final_reward": 0.0,
            "optimized_score": missing_score,
            "base_score": missing_score,
            "delta_score": missing_score,
            "deployed_score": missing_score,
            "deployed_base_score": missing_score,
            "deployed_delta_score": missing_score,
            "fallback_used": 0.0,
            "format_ok": 0.0,
            "optimized_prompt": "",
            "original_prompt": "",
            "biovil_alignment": missing_score,
            "label_consistency": missing_score,
            "raddino_realism": missing_score,
            "raddino_density": missing_score,
            "raddino_global_density": missing_score,
            "raddino_local_topk_density": missing_score,
            "raddino_memorization_penalty": missing_score,
            "raddino_nn_topk": missing_score,
            "raddino_nn_max": missing_score,
            "debug_image_path": "",
            "debug_base_image_path": "",
            "timing_s/prompt_parse_per_item": 0.0,
            "timing_s/sana_render_per_item": 0.0,
            "timing_s/sana_render_per_image": 0.0,
            "timing_s/reward_models_total": 0.0,
            "timing_s/reward_models_loop_per_item": 0.0,
            "timing_s/sana_batch_size": 0.0,
            "timing_s/online_reward_per_item": 0.0,
            "base_biovil_alignment": missing_score,
            "base_label_consistency": missing_score,
            "base_raddino_realism": missing_score,
            "base_raddino_density": missing_score,
            "base_raddino_global_density": missing_score,
            "base_raddino_local_topk_density": missing_score,
            "base_raddino_memorization_penalty": missing_score,
            "base_raddino_nn_topk": missing_score,
            "base_raddino_nn_max": missing_score,
            "base_timing_s/biovil": 0.0,
            "base_timing_s/label_classifier": 0.0,
            "base_timing_s/raddino": 0.0,
            "base_timing_s/reward_models_total": 0.0,
        }
        if "biovil" in self.reward.mode_parts:
            result["timing_s/biovil"] = 0.0
        if "label" in self.reward.mode_parts:
            result["timing_s/label_classifier"] = 0.0
        if "raddino" in self.reward.mode_parts:
            result["timing_s/raddino"] = 0.0
        return result

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        from diffusers import SanaPipeline

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "[llm_sana_reward] CUDA requested but unavailable; "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; "
                f"LLMSANA_REWARD_CUDA_VISIBLE_DEVICES={os.environ.get('LLMSANA_REWARD_CUDA_VISIBLE_DEVICES')!r}",
                flush=True,
            )
        if self.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device.startswith("cuda") and torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        local_files_only = os.environ.get("LLMSANA_LOCAL_FILES_ONLY", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        pipe = SanaPipeline.from_pretrained(
            self.sana_model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe = pipe.to(self.device)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(dtype)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.to(dtype)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return self._pipe

    def _render(self, prompt: str, seed: int):
        return self._render_batch([prompt], [seed])[0]

    def _render_batch(self, prompts: list[str], seeds: list[int]):
        pipe = self._load_pipe()
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generators = [torch.Generator(device=generator_device).manual_seed(seed) for seed in seeds]
        with torch.inference_mode():
            images = pipe(
                prompts,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generators,
            ).images
        return [image.convert("RGB") for image in images]

    def score_response(
        self,
        *,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any] | None = None,
        max_prompt_chars: int = 1200,
    ) -> dict[str, Any]:
        return self.score_batch(
            [
                {
                    "solution_str": solution_str,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                    "max_prompt_chars": max_prompt_chars,
                }
            ]
        )[0]

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_start = time.perf_counter()
        parse_start = time.perf_counter()
        results: list[dict[str, Any] | None] = [None for _ in requests]
        valid_items: list[dict[str, Any]] = []
        fallback_items: list[dict[str, Any]] = []

        for index, request in enumerate(requests):
            gt = _normalize_ground_truth(request.get("ground_truth"), request.get("extra_info"))
            original_prompt = str(gt.get("original_prompt", "")).strip()
            labels = gt.get("chexpert_labels", {})
            row_id = str(gt.get("row_id", ""))
            seed_material = f"{self.seed}:{row_id}:{original_prompt}"
            seed = int(hashlib.sha1(seed_material.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            optimized_prompt = extract_optimized_prompt(
                str(request.get("solution_str", "")),
                max_chars=int(request.get("max_prompt_chars", 1200)),
            )

            if not optimized_prompt:
                empty = self._empty_result()
                empty["original_prompt"] = original_prompt
                results[index] = empty
                fallback_items.append(
                    {
                        "index": index,
                        "row_id": row_id,
                        "original_prompt": original_prompt,
                        "labels": labels,
                        "seed": seed,
                    }
                )
                continue

            valid_items.append(
                {
                    "index": index,
                    "row_id": row_id,
                    "original_prompt": original_prompt,
                    "labels": labels,
                    "optimized_prompt": optimized_prompt,
                    "seed": seed,
                }
            )

        parse_time = time.perf_counter() - parse_start
        if not valid_items and not fallback_items:
            total_time = time.perf_counter() - total_start
            final_results = [result for result in results if result is not None]
            for result in final_results:
                result["timing_s/prompt_parse_per_item"] = parse_time / max(1, len(requests))
                result["timing_s/online_reward_per_item"] = total_time / max(1, len(requests))
                result["timing_s/sana_batch_size"] = 0.0
            return final_results

        with self._lock:
            render_start = time.perf_counter()
            render_prompts = (
                [item["optimized_prompt"] for item in valid_items]
                + [item["original_prompt"] for item in valid_items]
                + [item["original_prompt"] for item in fallback_items]
            )
            render_seeds = (
                [item["seed"] for item in valid_items]
                + [item["seed"] for item in valid_items]
                + [item["seed"] for item in fallback_items]
            )
            rendered_images = self._render_batch(render_prompts, render_seeds)
            render_time = time.perf_counter() - render_start
            optimized_images = rendered_images[: len(valid_items)]
            base_start = len(valid_items)
            base_end = base_start + len(valid_items)
            base_images = rendered_images[base_start:base_end]
            fallback_images = rendered_images[base_end:]
            reward_models_start = time.perf_counter()
            optimized_scored = [
                self.reward.score(image, item["original_prompt"], item["labels"])
                for image, item in zip(optimized_images, valid_items, strict=True)
            ]
            base_scored = [
                self.reward.score(image, item["original_prompt"], item["labels"])
                for image, item in zip(base_images, valid_items, strict=True)
            ]
            fallback_scored = [
                self.reward.score(image, item["original_prompt"], item["labels"])
                for image, item in zip(fallback_images, fallback_items, strict=True)
            ]
            reward_models_loop_time = time.perf_counter() - reward_models_start

        n_rendered_items = len(valid_items) + len(fallback_items)
        for image, base_image, item, result, base_result in zip(
            optimized_images, base_images, valid_items, optimized_scored, base_scored, strict=True
        ):
            optimized_score = float(result.get("score", 0.0))
            base_score = float(base_result.get("score", 0.0))
            delta_score = optimized_score - base_score
            relative_reward = _relative_prompt_reward(delta_score)
            if self.debug_image_dir:
                self.debug_image_dir.mkdir(parents=True, exist_ok=True)
                image_path = self.debug_image_dir / _safe_filename(f"{item['row_id']}_{item['optimized_prompt']}")
                image.save(image_path)
                result["debug_image_path"] = str(image_path)
                base_image_path = self.debug_image_dir / _safe_filename(f"{item['row_id']}_base_{item['original_prompt']}")
                base_image.save(base_image_path)
                result["debug_base_image_path"] = str(base_image_path)
            else:
                result["debug_image_path"] = ""
                result["debug_base_image_path"] = ""

            result["score"] = relative_reward
            result["final_reward"] = relative_reward
            result["optimized_score"] = optimized_score
            result["base_score"] = base_score
            result["delta_score"] = delta_score
            result["deployed_score"] = optimized_score
            result["deployed_base_score"] = base_score
            result["deployed_delta_score"] = delta_score
            result["fallback_used"] = 0.0
            for key, value in base_result.items():
                if isinstance(value, (int, float)):
                    result[f"base_{key}"] = float(value)
            result["format_ok"] = 1.0
            result["optimized_prompt"] = item["optimized_prompt"]
            result["original_prompt"] = item["original_prompt"]
            result["timing_s/prompt_parse_per_item"] = parse_time / max(1, len(requests))
            result["timing_s/sana_render_per_item"] = render_time / max(1, n_rendered_items)
            result["timing_s/sana_render_per_image"] = render_time / max(1, len(rendered_images))
            result["timing_s/reward_models_loop_per_item"] = reward_models_loop_time / max(1, n_rendered_items)
            result["timing_s/sana_batch_size"] = float(len(rendered_images))
            results[item["index"]] = result

        for base_image, item, result, base_result in zip(
            fallback_images, fallback_items, [results[item["index"]] for item in fallback_items], fallback_scored, strict=True
        ):
            if result is None:
                result = self._empty_result()
            base_score = float(base_result.get("score", 0.0))
            if self.debug_image_dir:
                self.debug_image_dir.mkdir(parents=True, exist_ok=True)
                base_image_path = self.debug_image_dir / _safe_filename(f"{item['row_id']}_fallback_{item['original_prompt']}")
                base_image.save(base_image_path)
                result["debug_base_image_path"] = str(base_image_path)
            else:
                result["debug_base_image_path"] = ""
            result["debug_image_path"] = ""
            result["score"] = 0.0
            result["final_reward"] = 0.0
            result["optimized_score"] = float("nan")
            result["base_score"] = float("nan")
            result["delta_score"] = float("nan")
            result["deployed_score"] = base_score
            result["deployed_base_score"] = base_score
            result["deployed_delta_score"] = 0.0
            result["fallback_used"] = 1.0
            result["format_ok"] = 0.0
            result["optimized_prompt"] = ""
            result["original_prompt"] = item["original_prompt"]
            result["timing_s/prompt_parse_per_item"] = parse_time / max(1, len(requests))
            result["timing_s/sana_render_per_item"] = render_time / max(1, n_rendered_items)
            result["timing_s/sana_render_per_image"] = render_time / max(1, len(rendered_images))
            result["timing_s/reward_models_loop_per_item"] = reward_models_loop_time / max(1, n_rendered_items)
            result["timing_s/sana_batch_size"] = float(len(rendered_images))
            results[item["index"]] = result

        total_time = time.perf_counter() - total_start
        final_results = [
            result
            if result is not None
            else {
                **self._empty_result(),
                "timing_s/prompt_parse_per_item": parse_time / max(1, len(requests)),
                "timing_s/sana_render_per_item": 0.0,
                "timing_s/sana_render_per_image": 0.0,
                "timing_s/reward_models_total": 0.0,
                "timing_s/online_reward_per_item": total_time / max(1, len(requests)),
                "timing_s/sana_batch_size": float(len(rendered_images)),
            }
            for result in results
        ]
        for result in final_results:
            result["timing_s/online_reward_per_item"] = total_time / max(1, len(requests))
        return final_results


def _make_key(
    *,
    sana_model_path: str | None,
    reward_mode: str,
    device: str,
    biovil_t_path: str | None,
    classifier_checkpoint: str | None,
    raddino_path: str | None,
    raddino_reference_cache: str | None,
    biovil_weight: float,
    label_weight: float,
    raddino_weight: float,
    raddino_score_mode: str,
    raddino_topk: int,
    raddino_density_temperature: float,
    raddino_cov_regularization: float,
    raddino_global_density_weight: float,
    raddino_local_topk_weight: float,
    raddino_memorization_weight: float,
    raddino_memorization_threshold: float,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    debug_image_dir: str | None,
) -> _OnlineRewardKey:
    return _OnlineRewardKey(
        sana_model_path=str(sana_model_path or _env_or_default("SANA_MODEL_PATH", _DEFAULT_SANA_MODEL)),
        reward_mode=reward_mode,
        device=device,
        biovil_t_path=str(biovil_t_path or _env_or_default("BIOVIL_T_PATH", _DEFAULT_BIOVIL)),
        classifier_checkpoint=str(classifier_checkpoint or _env_or_default("CXR_CLASSIFIER_CHECKPOINT", _DEFAULT_CLASSIFIER)),
        raddino_path=str(raddino_path or _env_or_default("RAD_DINO_PATH", _DEFAULT_RADDINO)),
        raddino_reference_cache=str(raddino_reference_cache or _env_or_default("RADDINO_REFERENCE_CACHE", _DEFAULT_RADDINO_CACHE)),
        biovil_weight=float(biovil_weight),
        label_weight=float(label_weight),
        raddino_weight=float(raddino_weight),
        raddino_score_mode=str(raddino_score_mode),
        raddino_topk=int(raddino_topk),
        raddino_density_temperature=float(raddino_density_temperature),
        raddino_cov_regularization=float(raddino_cov_regularization),
        raddino_global_density_weight=float(raddino_global_density_weight),
        raddino_local_topk_weight=float(raddino_local_topk_weight),
        raddino_memorization_weight=float(raddino_memorization_weight),
        raddino_memorization_threshold=float(raddino_memorization_threshold),
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        debug_image_dir=str(debug_image_dir or ""),
    )


def _get_engine(
    *,
    sana_model_path: str | None = None,
    reward_mode: str = "biovil_label_raddino",
    device: str | None = None,
    biovil_t_path: str | None = None,
    classifier_checkpoint: str | None = None,
    raddino_path: str | None = None,
    raddino_reference_cache: str | None = None,
    biovil_weight: float = 0.45,
    label_weight: float = 0.10,
    raddino_weight: float = 0.45,
    raddino_score_mode: str = "composite",
    raddino_topk: int = 8,
    raddino_density_temperature: float = 1.0,
    raddino_cov_regularization: float = 1e-3,
    raddino_global_density_weight: float = 0.60,
    raddino_local_topk_weight: float = 0.30,
    raddino_memorization_weight: float = 0.10,
    raddino_memorization_threshold: float = 0.97,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    seed: int = 42,
    debug_image_dir: str | None = None,
) -> LlmSanaOnlineReward:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = _make_key(
        sana_model_path=sana_model_path,
        reward_mode=reward_mode,
        device=device,
        biovil_t_path=biovil_t_path,
        classifier_checkpoint=classifier_checkpoint,
        raddino_path=raddino_path,
        raddino_reference_cache=raddino_reference_cache,
        biovil_weight=biovil_weight,
        label_weight=label_weight,
        raddino_weight=raddino_weight,
        raddino_score_mode=raddino_score_mode,
        raddino_topk=raddino_topk,
        raddino_density_temperature=raddino_density_temperature,
        raddino_cov_regularization=raddino_cov_regularization,
        raddino_global_density_weight=raddino_global_density_weight,
        raddino_local_topk_weight=raddino_local_topk_weight,
        raddino_memorization_weight=raddino_memorization_weight,
        raddino_memorization_threshold=raddino_memorization_threshold,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        debug_image_dir=debug_image_dir,
    )
    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            reward_config = CheXGenBenchRewardConfig(
                mode=key.reward_mode,
                biovil_weight=key.biovil_weight,
                label_weight=key.label_weight,
                raddino_weight=key.raddino_weight,
                raddino_score_mode=key.raddino_score_mode,
                raddino_topk=key.raddino_topk,
                raddino_density_temperature=key.raddino_density_temperature,
                raddino_cov_regularization=key.raddino_cov_regularization,
                raddino_global_density_weight=key.raddino_global_density_weight,
                raddino_local_topk_weight=key.raddino_local_topk_weight,
                raddino_memorization_weight=key.raddino_memorization_weight,
                raddino_memorization_threshold=key.raddino_memorization_threshold,
                biovil_t_path=key.biovil_t_path,
                classifier_checkpoint=key.classifier_checkpoint,
                raddino_path=key.raddino_path,
                raddino_reference_cache=key.raddino_reference_cache,
                device=key.device,
                local_files_only=os.environ.get("LLMSANA_LOCAL_FILES_ONLY", "0").lower()
                in {"1", "true", "yes"},
            )
            engine = LlmSanaOnlineReward(
                sana_model_path=key.sana_model_path,
                reward_config=reward_config,
                height=key.height,
                width=key.width,
                num_inference_steps=key.num_inference_steps,
                guidance_scale=key.guidance_scale,
                seed=key.seed,
                debug_image_dir=key.debug_image_dir,
            )
            _ENGINE_CACHE[key] = engine
    return engine


def _compute_score_inprocess(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    sana_model_path: str | None = None,
    reward_mode: str = "biovil_label_raddino",
    device: str | None = None,
    biovil_t_path: str | None = None,
    classifier_checkpoint: str | None = None,
    raddino_path: str | None = None,
    raddino_reference_cache: str | None = None,
    biovil_weight: float = 0.45,
    label_weight: float = 0.10,
    raddino_weight: float = 0.45,
    raddino_score_mode: str = "composite",
    raddino_topk: int = 8,
    raddino_density_temperature: float = 1.0,
    raddino_cov_regularization: float = 1e-3,
    raddino_global_density_weight: float = 0.60,
    raddino_local_topk_weight: float = 0.30,
    raddino_memorization_weight: float = 0.10,
    raddino_memorization_threshold: float = 0.97,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    seed: int = 42,
    debug_image_dir: str | None = None,
    max_prompt_chars: int = 1200,
    print_denominator: int = 64,
    max_print_chars: int = 2000,
) -> dict[str, Any]:
    engine = _get_engine(
        sana_model_path=sana_model_path,
        reward_mode=reward_mode,
        device=device,
        biovil_t_path=biovil_t_path,
        classifier_checkpoint=classifier_checkpoint,
        raddino_path=raddino_path,
        raddino_reference_cache=raddino_reference_cache,
        biovil_weight=biovil_weight,
        label_weight=label_weight,
        raddino_weight=raddino_weight,
        raddino_score_mode=raddino_score_mode,
        raddino_topk=raddino_topk,
        raddino_density_temperature=raddino_density_temperature,
        raddino_cov_regularization=raddino_cov_regularization,
        raddino_global_density_weight=raddino_global_density_weight,
        raddino_local_topk_weight=raddino_local_topk_weight,
        raddino_memorization_weight=raddino_memorization_weight,
        raddino_memorization_threshold=raddino_memorization_threshold,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        debug_image_dir=debug_image_dir,
    )
    result = engine.score_response(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        max_prompt_chars=max_prompt_chars,
    )

    should_print = print_denominator > 0 and random.randint(1, print_denominator) == print_denominator
    if should_print:
        _print_sampled_rollout(
            result=result,
            response=solution_str,
            max_print_chars=max_print_chars,
        )

    return result


def _print_sampled_rollout(
    *,
    result: dict[str, Any],
    response: str,
    max_print_chars: int,
) -> None:
    def _one_line(value: Any) -> str:
        text = str(value or "")
        clipped = text[:max_print_chars]
        return repr(clipped)

    print("=" * 80, flush=True)
    print("[llm_sana_reward] sampled rollout", flush=True)
    print(f"[score] {result.get('score')}", flush=True)
    print(f"[final_reward] {result.get('final_reward')}", flush=True)
    print(f"[optimized_score] {result.get('optimized_score')}", flush=True)
    print(f"[base_score] {result.get('base_score')}", flush=True)
    print(f"[delta_score] {result.get('delta_score')}", flush=True)
    print(f"[deployed_score] {result.get('deployed_score')}", flush=True)
    print(f"[deployed_base_score] {result.get('deployed_base_score')}", flush=True)
    print(f"[deployed_delta_score] {result.get('deployed_delta_score')}", flush=True)
    print(f"[fallback_used] {result.get('fallback_used')}", flush=True)
    for key in (
        "biovil_alignment",
        "label_consistency",
        "raddino_realism",
        "raddino_density",
        "raddino_global_density",
        "raddino_local_topk_density",
        "raddino_memorization_penalty",
        "raddino_nn_topk",
        "raddino_nn_max",
        "format_ok",
    ):
        if key in result:
            print(f"[{key}] {result[key]}", flush=True)
    original_prompt = str(result.get("original_prompt", "") or "")
    optimized_prompt = str(result.get("optimized_prompt", "") or "")
    response_text = str(response or "")
    print(f"[original_prompt len={len(original_prompt)}] {_one_line(original_prompt)}", flush=True)
    print(f"[optimized_prompt len={len(optimized_prompt)}] {_one_line(optimized_prompt)}", flush=True)
    print(f"[response len={len(response_text)}] {_one_line(response_text)}", flush=True)
    print("=" * 80, flush=True)


def _compute_score_batch_inprocess(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not payloads:
        return []
    if len(payloads) == 1:
        return [_compute_score_inprocess(**payloads[0])]

    first = payloads[0]
    engine = _get_engine(
        sana_model_path=first.get("sana_model_path"),
        reward_mode=first.get("reward_mode", "biovil_label_raddino"),
        device=first.get("device"),
        biovil_t_path=first.get("biovil_t_path"),
        classifier_checkpoint=first.get("classifier_checkpoint"),
        raddino_path=first.get("raddino_path"),
        raddino_reference_cache=first.get("raddino_reference_cache"),
        biovil_weight=first.get("biovil_weight", 0.45),
        label_weight=first.get("label_weight", 0.10),
        raddino_weight=first.get("raddino_weight", 0.45),
        raddino_score_mode=first.get("raddino_score_mode", "composite"),
        raddino_topk=first.get("raddino_topk", 8),
        raddino_density_temperature=first.get("raddino_density_temperature", 1.0),
        raddino_cov_regularization=first.get("raddino_cov_regularization", 1e-3),
        raddino_global_density_weight=first.get("raddino_global_density_weight", 0.60),
        raddino_local_topk_weight=first.get("raddino_local_topk_weight", 0.30),
        raddino_memorization_weight=first.get("raddino_memorization_weight", 0.10),
        raddino_memorization_threshold=first.get("raddino_memorization_threshold", 0.97),
        height=first.get("height", 512),
        width=first.get("width", 512),
        num_inference_steps=first.get("num_inference_steps", 20),
        guidance_scale=first.get("guidance_scale", 4.5),
        seed=first.get("seed", 42),
        debug_image_dir=first.get("debug_image_dir"),
    )
    results = engine.score_batch(
        [
            {
                "solution_str": payload.get("solution_str", ""),
                "ground_truth": payload.get("ground_truth"),
                "extra_info": payload.get("extra_info"),
                "max_prompt_chars": payload.get("max_prompt_chars", 1200),
            }
            for payload in payloads
        ]
    )

    for payload, result in zip(payloads, results, strict=True):
        print_denominator = int(payload.get("print_denominator", 64))
        max_print_chars = int(payload.get("max_print_chars", 2000))
        should_print = print_denominator > 0 and random.randint(1, print_denominator) == print_denominator
        if should_print:
            _print_sampled_rollout(
                result=result,
                response=str(payload.get("solution_str", "")),
                max_print_chars=max_print_chars,
            )

    return results


def _get_subprocess_client(cuda_visible_devices: str) -> RewardSubprocessClient:
    key = str(cuda_visible_devices)
    with _SUBPROCESS_CLIENTS_LOCK:
        client = _SUBPROCESS_CLIENTS.get(key)
        if client is None:
            client = RewardSubprocessClient(key)
            _SUBPROCESS_CLIENTS[key] = client
        return client


def _ray_actor_prefix(devices: list[str]) -> str:
    material = ":".join(
        [
            str(_PROJECT_ROOT),
            ",".join(devices),
            os.environ.get("LLMSANA_SANA_BATCH_SIZE", ""),
            os.environ.get("LLMSANA_SANA_BATCH_MAX_WAIT_MS", ""),
            os.environ.get("LLMSANA_REWARD_GPU_WORKER_MAX_CONCURRENCY", ""),
        ]
    )
    digest = hashlib.sha1(material.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"llm_sana_reward_{digest}"


def _get_ray_reward_actors(devices: list[str]):
    global _RAY_REWARD_SCHEDULER
    import ray

    with _RAY_REWARD_ACTORS_LOCK:
        if _RAY_REWARD_SCHEDULER is not None:
            return _RAY_REWARD_SCHEDULER, _RAY_REWARD_WORKERS

        prefix = _ray_actor_prefix(devices)
        sana_batch_size = max(1, int(os.environ.get("LLMSANA_SANA_BATCH_SIZE", "1")))
        sana_batch_max_wait_ms = max(0, int(os.environ.get("LLMSANA_SANA_BATCH_MAX_WAIT_MS", "50")))
        worker_max_concurrency = max(
            1,
            int(os.environ.get("LLMSANA_REWARD_GPU_WORKER_MAX_CONCURRENCY", str(max(8, sana_batch_size * 4)))),
        )
        scheduler_cls = ray.remote(num_cpus=0)(_RayRewardScheduler)
        worker_cls = ray.remote(num_cpus=0)(_RayRewardGpuWorker)

        _RAY_REWARD_SCHEDULER = scheduler_cls.options(
            name=f"{prefix}_scheduler",
            get_if_exists=True,
        ).remote(devices)

        for index, device in enumerate(devices):
            _RAY_REWARD_WORKERS[index] = worker_cls.options(
                name=f"{prefix}_gpu_{index}",
                get_if_exists=True,
                max_concurrency=worker_max_concurrency,
            ).remote(device, sana_batch_size, sana_batch_max_wait_ms)

        return _RAY_REWARD_SCHEDULER, _RAY_REWARD_WORKERS


def _compute_score_via_ray_dispatcher(payload: dict[str, Any], cuda_visible_devices: str) -> dict[str, Any]:
    import ray

    devices = _split_cuda_visible_devices(cuda_visible_devices)
    if not devices:
        return _compute_score_inprocess(**payload)

    scheduler, workers = _get_ray_reward_actors(devices)
    dispatch_start = time.perf_counter()
    worker_index, selected_device, inflight = ray.get(scheduler.acquire.remote())
    print(
        "[llm_sana_reward] dispatch reward request; "
        f"pool={cuda_visible_devices!r}; selected={selected_device!r}; inflight={inflight}",
        flush=True,
    )
    try:
        result = ray.get(workers[worker_index].compute.remote(payload))
        result["timing_s/reward_dispatch_total"] = time.perf_counter() - dispatch_start
        return result
    finally:
        scheduler.release.remote(worker_index)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    sana_model_path: str | None = None,
    reward_mode: str = "biovil_label_raddino",
    device: str | None = None,
    biovil_t_path: str | None = None,
    classifier_checkpoint: str | None = None,
    raddino_path: str | None = None,
    raddino_reference_cache: str | None = None,
    biovil_weight: float = 0.45,
    label_weight: float = 0.10,
    raddino_weight: float = 0.45,
    raddino_score_mode: str = "composite",
    raddino_topk: int = 8,
    raddino_density_temperature: float = 1.0,
    raddino_cov_regularization: float = 1e-3,
    raddino_global_density_weight: float = 0.60,
    raddino_local_topk_weight: float = 0.30,
    raddino_memorization_weight: float = 0.10,
    raddino_memorization_threshold: float = 0.97,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    seed: int = 42,
    debug_image_dir: str | None = None,
    max_prompt_chars: int = 1200,
    print_denominator: int = 64,
    max_print_chars: int = 2000,
    **_: Any,
) -> dict[str, Any]:
    """verl-compatible custom reward for text-policy GRPO."""
    payload = {
        "data_source": data_source,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "sana_model_path": sana_model_path,
        "reward_mode": reward_mode,
        "device": device,
        "biovil_t_path": biovil_t_path,
        "classifier_checkpoint": classifier_checkpoint,
        "raddino_path": raddino_path,
        "raddino_reference_cache": raddino_reference_cache,
        "biovil_weight": biovil_weight,
        "label_weight": label_weight,
        "raddino_weight": raddino_weight,
        "raddino_score_mode": raddino_score_mode,
        "raddino_topk": raddino_topk,
        "raddino_density_temperature": raddino_density_temperature,
        "raddino_cov_regularization": raddino_cov_regularization,
        "raddino_global_density_weight": raddino_global_density_weight,
        "raddino_local_topk_weight": raddino_local_topk_weight,
        "raddino_memorization_weight": raddino_memorization_weight,
        "raddino_memorization_threshold": raddino_memorization_threshold,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "debug_image_dir": debug_image_dir,
        "max_prompt_chars": max_prompt_chars,
        "print_denominator": print_denominator,
        "max_print_chars": max_print_chars,
    }
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_cuda_visible_devices = os.environ.get("LLMSANA_REWARD_CUDA_VISIBLE_DEVICES")
    use_dispatcher = os.environ.get("LLMSANA_REWARD_USE_DISPATCHER", "1").lower() not in {"0", "false", "no"}
    if resolved_device.startswith("cuda") and raw_cuda_visible_devices and use_dispatcher:
        return _compute_score_via_ray_dispatcher(payload, raw_cuda_visible_devices)

    cuda_visible_devices = _select_reward_cuda_visible_devices()
    if resolved_device.startswith("cuda") and cuda_visible_devices:
        print(
            "[llm_sana_reward] running reward in CUDA subprocess; "
            f"parent CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; "
            f"subprocess CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}",
            flush=True,
        )
        return _get_subprocess_client(cuda_visible_devices).compute(payload)
    return _compute_score_inprocess(**payload)
