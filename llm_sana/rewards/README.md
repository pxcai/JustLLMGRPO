# CheXGenBench Reward

This reward is for the LLM-planner/Sana-renderer GRPO setup.

Available modes:

- `biovil`: BioViL-T image-text alignment.
- `biovil_label`: BioViL-T alignment + CheXpert label consistency.
- `biovil_label_raddino`: BioViL-T alignment + label consistency + RadDINO realism proxy.

Paper reward weights:

```text
score = 0.45 * BioViL-T alignment
      + 0.10 * label consistency
      + 0.45 * RadDINO fidelity
```

Default RadDINO realism proxy:

```text
raddino_realism_i =
      0.60 * global_density_i
    + 0.30 * local_topk_density_i
    - 0.10 * memorization_penalty_i
```

Build the RadDINO real-image reference cache once:

```bash
cd /path/to/JustLLMGRPO
bash scripts/build_raddino_cache.sh
```

Use from Python:

```python
from llm_sana.rewards import CheXGenBenchReward, CheXGenBenchRewardConfig

reward = CheXGenBenchReward(
    CheXGenBenchRewardConfig(
        mode="biovil_label_raddino",
        biovil_t_path="microsoft/BiomedVLP-BioViL-T",
        classifier_checkpoint="artifacts/best_classifier.pt",
        raddino_path="microsoft/rad-dino",
        raddino_reference_cache="artifacts/raddino_train20k_ref.npz",
    )
)

result = reward.score(generated_image_path, original_annotated_prompt, chexpert_labels)
```

Run JustLLMGRPO training through the release launcher:

```bash
cd /path/to/JustLLMGRPO
bash scripts/train_justllmgrpo.sh
```

Switch rewards without changing code:

```bash
REWARD_MODE=biovil bash llm_sana/run_verl_qwen3_llm_sana_grpo.sh
REWARD_MODE=biovil_label bash llm_sana/run_verl_qwen3_llm_sana_grpo.sh
REWARD_MODE=biovil_label_raddino bash llm_sana/run_verl_qwen3_llm_sana_grpo.sh
```

The GRPO flow is text-policy based:

1. The parquet `prompt` is built from `LLAVARAD_ANNOTATIONS_TRAIN.csv` `annotated_prompt`.
2. Qwen outputs an optimized diffusion prompt.
3. The custom reward loads frozen Sana, renders one image, and scores the image.
4. verl receives `score` plus component metrics such as `biovil_alignment`, `label_consistency`, and `raddino_realism`.
