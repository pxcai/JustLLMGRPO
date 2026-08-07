# Reproducibility protocol

## Training sequence

1. Obtain the LLaVA-Rad/CheXGenBench annotation splits and MIMIC-CXR-JPG images.
2. Install the pinned environment and patched VERL checkout.
3. Train or supply the frozen 14-label CXR classifier.
4. Build the real-image RadDINO reference cache.
5. Set machine-specific paths in `configs/paths.env`.
6. Run `scripts/train_justllmgrpo.sh` for 400 optimizer steps.
7. Select and merge the desired actor checkpoint with `scripts/merge_checkpoint.sh`.
8. Run `scripts/infer.sh` with the merged policy and frozen Sana.

## Paper configuration

The exact scalar settings are stored in `configs/paper_experiment.env`. The launcher does not shuffle the training parquet, uses one GRPO update epoch and one optimizer step per rollout batch, and updates all Qwen parameters. A frozen initial Qwen copy supplies KL regularization.

For every valid candidate prompt, the reward worker:

1. derives a deterministic seed from the global seed, row identifier, and source prompt;
2. renders the candidate prompt and source prompt with the same initial noise;
3. evaluates both images against the source prompt and CheXpert labels;
4. converts the candidate-minus-source score difference into the relative reward;
5. returns the reward and diagnostic components to VERL.

For score difference $\Delta$, the released reward is $1+\Delta$ when $\Delta>0$, $0.1$ when $-0.03\leq\Delta\leq0$, and $0$ otherwise. GRPO standardizes rewards within the five candidates sampled for each source prompt.

The paper reward weights are 0.45 BioViL-T, 0.10 label consistency, and 0.45 RadDINO fidelity. The RadDINO component uses 0.60 global-density agreement, 0.30 local top-k similarity, and a 0.10 memorization penalty.

## Hardware

The reported allocation used five 80-GB GPUs:

- policy rollout and FSDP optimization: GPU 0 and GPU 1;
- frozen Sana rendering and reward computation: GPU 2, GPU 3, and GPU 4.

The topology is controlled by `LLM_CUDA_VISIBLE_DEVICES` and `REWARD_CUDA_VISIBLE_DEVICES`. Smaller GPUs may require reduced rollout batch size, rollout count, Sana batch size, and vLLM memory utilization.

## Determinism

The code fixes Python, NumPy, and Torch seeds where applicable. CUDA kernels, distributed scheduling, vLLM, and model-library changes may still introduce numerical variation. The paper reports one training run and does not claim statistical significance.

## Expected artifacts

Training produces distributed FSDP checkpoints. These are not directly loadable by standard Hugging Face inference. Run the merger script and use the resulting `merged_hf/global_step_<N>` directory as `PLANNER_MODEL`.
