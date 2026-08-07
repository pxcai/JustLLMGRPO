"""CheXGenBench-aligned reward for GRPO prompt planning.

This module is designed for the LLM-planner/Sana-renderer setup:

1. The LLM rewrites an original LLaVA-Rad/MIMIC caption into a diffusion prompt.
2. A frozen diffusion renderer generates a chest radiograph.
3. This reward scores the generated image against the original caption and labels.

The reward mode is configurable:

- ``biovil``: BioViL-T image-text alignment only.
- ``biovil_label``: BioViL-T plus CXR multi-label consistency.
- ``biovil_label_raddino``: BioViL-T plus label consistency plus RadDINO realism.

All heavyweight models are loaded lazily, only when their component is enabled.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


MIMIC_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


@dataclass
class CheXGenBenchRewardConfig:
    """Runtime config for CheXGenBench-style reward."""

    mode: str = "biovil_label_raddino"
    biovil_weight: float = 0.45
    label_weight: float = 0.10
    raddino_weight: float = 0.45
    raddino_score_mode: str = "composite"
    raddino_topk: int = 8
    raddino_density_temperature: float = 1.0
    raddino_cov_regularization: float = 1e-3
    raddino_global_density_weight: float = 0.60
    raddino_local_topk_weight: float = 0.30
    raddino_memorization_weight: float = 0.10
    raddino_memorization_threshold: float = 0.97

    biovil_t_path: str | None = None
    classifier_checkpoint: str | None = None
    raddino_path: str | None = None
    raddino_reference_cache: str | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_model_name: str = "resnet50"
    classifier_threshold: float = 0.5
    raddino_max_reference_images: int = 20000
    clamp_component_scores: bool = True
    local_files_only: bool = False


def _as_path(value: str | os.PathLike[str] | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _existing_local_path(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _parse_labels(labels: dict[str, Any] | str | None) -> dict[str, Any]:
    if labels is None:
        return {}
    if isinstance(labels, dict):
        return labels
    return ast.literal_eval(labels)


@contextlib.contextmanager
def _image_path(image: str | os.PathLike[str] | Image.Image):
    """Yield a filesystem path for either an existing path or a PIL image."""
    if isinstance(image, str | os.PathLike):
        yield Path(image)
        return

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        image.save(tmp_path)
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


class _MultiLabelClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=False)
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "classifier"):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "head"):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            in_features = self.model.get_classifier().in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CheXGenBenchReward:
    """Composable CheXGenBench-aligned reward.

    Example:
        >>> reward = CheXGenBenchReward(
        ...     CheXGenBenchRewardConfig(
        ...         mode="biovil_label_raddino",
        ...         biovil_t_path="microsoft/BiomedVLP-BioViL-T",
        ...         classifier_checkpoint="evaluate_chexgenbench/outputs/sana_e20/downstream_classification/results/best_classifier.pt",
        ...         raddino_path="microsoft/rad-dino",
        ...         raddino_reference_cache="llm_sana/cache/raddino_mimic_train.npz",
        ...     )
        ... )
        >>> out = reward.score("sample.png", "PA chest radiograph ...", labels)
    """

    def __init__(self, config: CheXGenBenchRewardConfig | None = None):
        self.config = config or CheXGenBenchRewardConfig()
        self.mode_parts = set(self.config.mode.split("_"))

        valid_modes = {
            "biovil",
            "label",
            "raddino",
            "biovil_label",
            "biovil_raddino",
            "label_raddino",
            "biovil_label_raddino",
        }
        if self.config.mode not in valid_modes:
            raise ValueError(f"Unsupported reward mode {self.config.mode!r}. Valid modes: {sorted(valid_modes)}")

        self._biovil_engine = None
        self._classifier = None
        self._classifier_transform = None
        self._classifier_label_names: list[str] | None = None
        self._raddino_model = None
        self._raddino_processor = None
        self._raddino_reference_features = None
        self._raddino_reference_mean = None
        self._raddino_reference_inv_var = None
        self._raddino_reference_feature_dim = None
        self._last_raddino_diagnostics: dict[str, float] = {}

    def score(
        self,
        image: str | os.PathLike[str] | Image.Image,
        prompt: str,
        labels: dict[str, Any] | str | None = None,
    ) -> dict[str, float]:
        """Score one generated image.

        Args:
            image: Generated image path or PIL image.
            prompt: Original clinical caption to preserve.
            labels: CheXpert-style label dict from CheXGenBench CSV.

        Returns:
            A dict with ``score`` plus enabled component scores.
        """
        total_start = time.perf_counter()
        out: dict[str, float] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        with _image_path(image) as path:
            if "biovil" in self.mode_parts:
                component_start = time.perf_counter()
                value = self.biovil_alignment(path, prompt)
                out["timing_s/biovil"] = time.perf_counter() - component_start
                out["biovil_alignment"] = value
                weighted_sum += self.config.biovil_weight * value
                weight_sum += self.config.biovil_weight

            if "label" in self.mode_parts:
                component_start = time.perf_counter()
                value = self.label_consistency(path, labels)
                out["timing_s/label_classifier"] = time.perf_counter() - component_start
                out["label_consistency"] = value
                weighted_sum += self.config.label_weight * value
                weight_sum += self.config.label_weight

            if "raddino" in self.mode_parts:
                component_start = time.perf_counter()
                value = self.raddino_realism(path)
                out["timing_s/raddino"] = time.perf_counter() - component_start
                out["raddino_realism"] = value
                out.update(self._last_raddino_diagnostics)
                weighted_sum += self.config.raddino_weight * value
                weight_sum += self.config.raddino_weight

        out["score"] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        out["timing_s/reward_models_total"] = time.perf_counter() - total_start
        return out

    def biovil_alignment(self, image_path: Path, prompt: str) -> float:
        engine = self._load_biovil_engine()
        score = float(engine.get_similarity_score_from_raw_data(image_path, prompt))
        return _clip01(score) if self.config.clamp_component_scores else score

    def label_consistency(self, image_path: Path, labels: dict[str, Any] | str | None) -> float:
        labels_dict = _parse_labels(labels)
        if not labels_dict:
            return 0.0

        model, transform, label_names = self._load_classifier()
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].detach().cpu().numpy()

        terms = []
        for idx, name in enumerate(label_names):
            raw = labels_dict.get(name)
            if raw is None:
                continue
            try:
                label = float(raw)
            except (TypeError, ValueError):
                continue
            if label == 1.0:
                terms.append(float(probs[idx]))
            elif label == 0.0:
                terms.append(float(1.0 - probs[idx]))
            # CheXpert uncertain labels (-1) are ignored.

        return float(np.mean(terms)) if terms else 0.0

    def raddino_realism(self, image_path: Path) -> float:
        refs, mean, inv_var, feature_dim = self._load_raddino_reference_stats()
        feat = self._raddino_features([image_path])[0]

        centered = feat - mean
        density_dist = float(np.sum(np.square(centered) * inv_var))
        density_scale = max(1.0, float(feature_dim) * float(self.config.raddino_density_temperature))
        density_score = float(np.exp(-0.5 * density_dist / density_scale))

        sims = refs @ feat
        topk = max(1, min(int(self.config.raddino_topk), int(sims.shape[0])))
        topk_vals = np.partition(sims, -topk)[-topk:]
        local_topk_density = _clip01((float(np.mean(topk_vals)) + 1.0) / 2.0)
        nn_max = _clip01((float(np.max(sims)) + 1.0) / 2.0)
        threshold = float(self.config.raddino_memorization_threshold)
        memorization_penalty = _clip01((nn_max - threshold) / max(1e-6, 1.0 - threshold))

        mode = str(self.config.raddino_score_mode).lower()
        if mode == "density":
            realism = density_score
        elif mode == "nearest":
            realism = nn_max
        elif mode == "topk":
            realism = local_topk_density
        elif mode == "hybrid":
            realism = 0.7 * density_score + 0.3 * local_topk_density
        elif mode == "composite":
            realism = (
                float(self.config.raddino_global_density_weight) * density_score
                + float(self.config.raddino_local_topk_weight) * local_topk_density
                - float(self.config.raddino_memorization_weight) * memorization_penalty
            )
        else:
            raise ValueError(
                f"Unsupported raddino_score_mode={self.config.raddino_score_mode!r}. "
                "Valid modes: composite, density, nearest, topk, hybrid."
            )

        # Preserve the legacy name for downstream logging, but add explicit
        # diagnostics so the reward is easier to interpret.
        self._last_raddino_diagnostics = {
            "raddino_density": density_score,
            "raddino_global_density": density_score,
            "raddino_local_topk_density": local_topk_density,
            "raddino_memorization_penalty": memorization_penalty,
            "raddino_nn_topk": local_topk_density,
            "raddino_nn_max": nn_max,
        }
        return _clip01(realism)

    def _load_biovil_engine(self):
        if self._biovil_engine is not None:
            return self._biovil_engine

        local_path = _existing_local_path(self.config.biovil_t_path)
        if local_path:
            image_weights = local_path / "biovil_t_image_model_proj_size_128.pt"
            if not image_weights.exists():
                raise FileNotFoundError(f"Missing BioViL-T image weights: {image_weights}")

            import health_multimodal.image.model.pretrained as image_pretrained
            import health_multimodal.text.utils as text_utils

            image_pretrained.BIOMED_VLP_BIOVIL_T = str(local_path)

            def local_biovil_t_image_weights():
                return image_weights

            image_pretrained._download_biovil_t_image_model_weights = local_biovil_t_image_weights

            def local_biovil_t_bert():
                tokenizer = text_utils.CXRBertTokenizer.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True
                )
                text_model = text_utils.CXRBertModel.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True
                )
                return tokenizer, text_model

            text_utils.get_biovil_t_bert = local_biovil_t_bert

        from health_multimodal.image import get_image_inference
        from health_multimodal.text import get_bert_inference
        from health_multimodal.vlp import ImageTextInferenceEngine

        text_inference = get_bert_inference()
        image_inference = get_image_inference()
        text_inference.model.to(self.config.device)
        image_inference.model.to(self.config.device)
        self._biovil_engine = ImageTextInferenceEngine(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
        )
        return self._biovil_engine

    def _load_classifier(self):
        if self._classifier is not None:
            return self._classifier, self._classifier_transform, self._classifier_label_names

        checkpoint_path = _as_path(self.config.classifier_checkpoint)
        if checkpoint_path is None:
            raise ValueError("classifier_checkpoint is required when reward mode includes 'label'.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        label_names = checkpoint.get("label_names", MIMIC_PATHOLOGIES)
        model_name = checkpoint.get("args", {}).get("model_name", self.config.classifier_model_name)

        model = _MultiLabelClassifier(model_name, num_classes=len(label_names))
        state = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(self.config.device)
        model.eval()

        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._classifier = model
        self._classifier_transform = transform
        self._classifier_label_names = list(label_names)
        return self._classifier, self._classifier_transform, self._classifier_label_names

    def _load_raddino(self):
        if self._raddino_model is not None:
            return self._raddino_model, self._raddino_processor

        raddino_ref = self.config.raddino_path
        if not raddino_ref:
            raise ValueError("raddino_path is required when reward mode includes 'raddino'.")
        from transformers import AutoImageProcessor, AutoModel

        self._raddino_model = AutoModel.from_pretrained(
            str(raddino_ref), local_files_only=self.config.local_files_only
        ).to(self.config.device)
        self._raddino_processor = AutoImageProcessor.from_pretrained(
            str(raddino_ref), local_files_only=self.config.local_files_only
        )
        self._raddino_model.eval()
        return self._raddino_model, self._raddino_processor

    def _raddino_features(self, image_paths: Iterable[Path], batch_size: int = 16) -> np.ndarray:
        model, processor = self._load_raddino()
        feats = []
        paths = list(image_paths)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_feat = outputs.pooler_output
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                batch_feat = outputs.image_embeds
            else:
                batch_feat = outputs.last_hidden_state[:, 0]
            batch_feat = torch.nn.functional.normalize(batch_feat.float(), dim=-1)
            feats.append(batch_feat.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _load_raddino_reference_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if (
            self._raddino_reference_features is not None
            and self._raddino_reference_mean is not None
            and self._raddino_reference_inv_var is not None
            and self._raddino_reference_feature_dim is not None
        ):
            return (
                self._raddino_reference_features,
                self._raddino_reference_mean,
                self._raddino_reference_inv_var,
                int(self._raddino_reference_feature_dim),
            )

        cache_path = _as_path(self.config.raddino_reference_cache)
        if cache_path is None:
            raise ValueError("raddino_reference_cache is required when reward mode includes 'raddino'.")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"RadDINO reference cache not found: {cache_path}. "
                "Build it with `python -m llm_sana.rewards.chexgenbench_reward build-raddino-cache ...`."
            )
        data = np.load(cache_path)
        feats = data["features"].astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-12, None)
        mean = feats.mean(axis=0)
        centered = feats - mean
        variance = centered.var(axis=0) + float(self.config.raddino_cov_regularization)
        inv_var = 1.0 / np.clip(variance, 1e-12, None)

        self._raddino_reference_features = feats
        self._raddino_reference_mean = mean.astype(np.float32)
        self._raddino_reference_inv_var = inv_var.astype(np.float32)
        self._raddino_reference_feature_dim = int(feats.shape[1])
        return feats, self._raddino_reference_mean, self._raddino_reference_inv_var, int(feats.shape[1])

    def build_raddino_reference_cache(
        self,
        real_csv: str | os.PathLike[str],
        real_image_dir: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        image_col: str = "path",
        caption_col: str = "annotated_prompt",
        max_images: int | None = None,
        batch_size: int = 16,
    ) -> Path:
        """Build a RadDINO feature cache from real CXR images."""
        import pandas as pd

        real_csv = Path(real_csv)
        real_image_dir = Path(real_image_dir)
        output_path = Path(output_path)
        max_images = max_images or self.config.raddino_max_reference_images

        df = pd.read_csv(real_csv)
        if caption_col in df.columns:
            df = df.drop_duplicates(subset=[caption_col]).reset_index(drop=True)
        if max_images > 0:
            df = df.head(max_images)
        paths = [real_image_dir / rel for rel in df[image_col].tolist()]
        paths = [path for path in paths if path.exists()]
        if not paths:
            raise ValueError(f"No real images found from {real_csv} under {real_image_dir}")

        features = self._raddino_features(paths, batch_size=batch_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, features=features, image_paths=np.array([str(p) for p in paths], dtype=object))
        return output_path


def _build_cache_cli(args) -> None:
    reward = CheXGenBenchReward(
        CheXGenBenchRewardConfig(
            mode="raddino",
            raddino_path=args.raddino_path,
            device=args.device,
        )
    )
    path = reward.build_raddino_reference_cache(
        real_csv=args.real_csv,
        real_image_dir=args.real_image_dir,
        output_path=args.output,
        image_col=args.image_col,
        caption_col=args.caption_col,
        max_images=args.max_images,
        batch_size=args.batch_size,
    )
    print(f"Saved RadDINO reference cache to {path}")


def _score_cli(args) -> None:
    reward = CheXGenBenchReward(
        CheXGenBenchRewardConfig(
            mode=args.mode,
            biovil_t_path=args.biovil_t_path,
            classifier_checkpoint=args.classifier_checkpoint,
            raddino_path=args.raddino_path,
            raddino_reference_cache=args.raddino_reference_cache,
            device=args.device,
            biovil_weight=args.biovil_weight,
            label_weight=args.label_weight,
            raddino_weight=args.raddino_weight,
        )
    )
    labels = Path(args.labels).read_text() if args.labels and Path(args.labels).exists() else args.labels
    print(reward.score(args.image, args.prompt, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="CheXGenBench-aligned reward utilities.")
    sub = parser.add_subparsers(required=True)

    cache = sub.add_parser("build-raddino-cache")
    cache.add_argument("--real_csv", required=True)
    cache.add_argument("--real_image_dir", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--raddino_path", required=True)
    cache.add_argument("--image_col", default="path")
    cache.add_argument("--caption_col", default="annotated_prompt")
    cache.add_argument("--max_images", type=int, default=20000)
    cache.add_argument("--batch_size", type=int, default=16)
    cache.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cache.set_defaults(func=_build_cache_cli)

    score = sub.add_parser("score")
    score.add_argument("--image", required=True)
    score.add_argument("--prompt", required=True)
    score.add_argument("--labels", default=None, help="Label dict string or path to a text file containing one.")
    score.add_argument("--mode", default="biovil_label_raddino")
    score.add_argument("--biovil_t_path", default=None)
    score.add_argument("--classifier_checkpoint", default=None)
    score.add_argument("--raddino_path", default=None)
    score.add_argument("--raddino_reference_cache", default=None)
    score.add_argument("--biovil_weight", type=float, default=0.45)
    score.add_argument("--label_weight", type=float, default=0.10)
    score.add_argument("--raddino_weight", type=float, default=0.45)
    score.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    score.set_defaults(func=_score_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
