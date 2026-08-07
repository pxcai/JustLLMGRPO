# Troubleshooting

## Model reference reported as missing

Model options accept either a local directory or a Hugging Face repository ID. If `LLMSANA_LOCAL_FILES_ONLY=1` is set, every model must already exist locally or in the Hugging Face cache.

## BioViL-T import failure

Install `hi-ml-multimodal==0.2.2` and verify:

```bash
python -c "import health_multimodal"
```

For an offline run, download `microsoft/BiomedVLP-BioViL-T` and point `BIOVIL_T_PATH` to the local snapshot.

## vLLM or CUDA out of memory

Reduce one or more of:

```bash
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
ROLLOUT_N=2
SANA_BATCH_SIZE=4
ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
```

For inference, use `PLANNER_BACKEND=transformers` or reduce `PLANNER_TP_SIZE`, `--planner-batch-size`, and `--sana-batch-size`.

## Ray reward workers cannot see the reward GPUs

Confirm that policy and reward device lists refer to physical device identifiers visible before the launcher changes `CUDA_VISIBLE_DEVICES`:

```bash
LLM_CUDA_VISIBLE_DEVICES=0,1
REWARD_CUDA_VISIBLE_DEVICES=2,3,4
```

The launcher forwards the reward device pool through Ray runtime environment variables.

## Invalid policy completions receive zero reward

A valid response must contain a closing `</think>` marker followed by one non-empty `<optimized_prompt>...</optimized_prompt>` block. This is intentional and matches inference parsing. The shared template is in `llm_sana/prompts.py`.

## FSDP checkpoint cannot be used for inference

Run:

```bash
RUN_DIR=outputs/<run-name> STEP=<step> bash scripts/merge_checkpoint.sh
```

Use the resulting `merged_hf/global_step_<step>` directory, not the distributed actor directory.
