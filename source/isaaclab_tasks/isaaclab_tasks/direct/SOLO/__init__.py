"""SOLO G1 AMP, PPO, and state-estimation environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-G1-AMP-Walk-SOLO-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-Dance-SOLO-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-SOLO-Direct-v0",
    entry_point=f"{__name__}.g1_ppo_env:G1PpoWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_ppo_env:G1PpoWalkEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_ppo_cfg.yaml",
    },
)

