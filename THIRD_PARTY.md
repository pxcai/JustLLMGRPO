# Third-party dependencies

This repository does not redistribute complete third-party repositories or model weights.

## VERL

- Repository: `https://github.com/verl-project/verl.git`
- Pinned revision: `4cd50e69b73b4ff0df750264f89e49c94c112c15`
- Local modifications: `third_party/verl_overrides/`
- Purpose: standard GRPO optimization, FSDP actor training, and vLLM rollout.

Copying the override tree onto the pinned checkout adds the LLM-Sana entrypoint and propagates asynchronous image-reward metrics through the rollout and trainer stack.

## Sana and reward models

The main method does not modify the Sana source tree. The frozen CXR-adapted
renderer is loaded through Diffusers from
`raman07/CheXGenBench-Models-Sana-e20`. BioViL-T and RadDINO are likewise
loaded from their public model repositories. Their model cards and licenses
apply independently of this repository.

All third-party files remain subject to their upstream licenses and attribution requirements.
