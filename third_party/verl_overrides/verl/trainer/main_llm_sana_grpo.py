# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""VERL entrypoint for LLM-Sana GRPO.

This entrypoint deliberately reuses VERL's PPO/GRPO trainer, actor workers, and
vLLM rollout server.  It is the stable path for training the LLM prompt policy
with Sana-rendered online rewards.

It does not update Sana.  Full joint LLM+Sana GRPO needs a dedicated Sana actor
worker group and trainer loop.
"""

import hydra

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)


if __name__ == "__main__":
    main()
