# SOLO G1

### State estimation from robot joint observations

SOLO G1 trains 29-DOF Unitree G1 walk and dance policies with an
[Adversarial Motion Prior (AMP)](https://xbpeng.github.io/projects/AMP/), then replaces policy-only base information
with a learned state estimator. The G1 environment and reference motions are adapted from
[`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp); the estimator/DAgger workflow follows
`SOLO_DEXTRA`.

The default estimator uses every G1 joint: 29 joint positions plus the 29 simulator joint velocities (58D). It predicts
root linear velocity, root angular velocity, and gravity/orientation state (9D). No finite-difference velocity,
encoder quantization, or EMA filtering is used.

## Pipeline

### Phase 1 — privileged policy

Train an AMP walk or dance teacher with the 101D G1 motion observation. Both environments combine their task reward
with the learned style reward. With SKRL 2.x, the independent default scales are
`task_reward_scale: 1.0` and `style_reward_scale: 2.0`; they are not weights that sum to one.
Physics runs at 120 Hz with decimation 4 (30 Hz policy control), and the discriminator receives four policy-spaced
AMP frames (4 x 101D = 404D). Episodes last 20 seconds. The walk task uses a 0.6 m/s reference-aligned target and
Dextra-style action-rate and normalized torque-saturation penalties; dance keeps posture/height rewards without a
velocity target.

```bash
# Walk AMP
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --algorithm AMP --headless

# Dance AMP
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-G1-AMP-Dance-SOLO-Direct-v0 --algorithm AMP --headless

# A policy-only SKRL PPO walking example using the same 29-DOF interface
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-G1-PPO-Walk-SOLO-Direct-v0 --algorithm PPO --headless
```

### Phase 2 — estimator and distillation

Collect frozen-teacher rollouts and train the default two-layer, hidden-256 LSTM with history 50. Add
`--joint-preset legs` or `--joint-preset upper` for the input ablations, or select `--estimator TCN`/`MLP`.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/train_state_estimator.py \
  --teacher-checkpoint <best_agent.pt> \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 \
  --estimator LSTM --window 50 --joint-preset all --dagger-rounds 10 --headless
```

The basic teacher-student baseline uses all 58 joint values and a `256-256-128` MLP to predict the teacher's 29D
action. `--dagger-rounds 0` is vanilla offline distillation; a positive value enables Student DAgger.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/train_distillation.py \
  --teacher-checkpoint <best_agent.pt> --dagger-rounds 0 --headless
```

### Phase 3 — estimated-state inference

Teacher checkpoints are loaded through SKRL's public `Runner`/`agent.load()` API. The player uses the SKRL 2.x
evaluation API and can produce an action/estimate CSV alongside a video.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/play_with_estimator.py \
  --teacher-checkpoint <best_agent.pt> --estimator-checkpoint <best_estimator.pt> \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --rollout-csv logs/rollout.csv
```

## Motion tools

The bundled `G1_walk.npz` and `G1_dance.npz` follow the 29-joint reference schema. Available tools include schema
validation, matplotlib visualization, Isaac Sim replay/recording, conversion, pelvis alignment, and reference tracking.

```bash
# Inspect schema and values
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/verify_motion.py \
  --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/G1_walk.npz

# Isaac Sim replay; --record-output writes another compatible NPZ
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/tools/replay_motion.py \
  --motion source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/G1_dance.npz
```

Motion-variance generators and Dextra-specific AX18/hardware/system-identification code are intentionally excluded.

## Ablation

The runner covers teacher, estimator architecture/history/joint presets, vanilla distillation, and Student DAgger for
AMP walk, AMP dance, and PPO walk. A failed run is recorded and the remaining matrix continues. Each task needs its
own teacher checkpoint mapping.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/run_ablation.py \
  --teacher-checkpoint amp_walk=<walk_amp.pt> \
  --teacher-checkpoint amp_dance=<dance_amp.pt> \
  --teacher-checkpoint ppo_walk=<walk_ppo.pt> \
  --seeds 42 43 44 --headless
```

Outputs include raw JSONL, aggregate JSON, tidy/summary CSV, Markdown and LaTeX tables, `report.md`, and PNG/PDF
plots with mean, standard deviation, and 95% confidence intervals. Estimator error, target-specific error, closed-loop
return/termination, action agreement/smoothness, dynamics/energy, AMP rewards, parameter count, and inference latency
are supported metric fields.

## SKRL compatibility

- Supported dependency: `skrl>=2.0,<3.0`.
- Configs use `observation_preprocessor`, `amp_observation_preprocessor`, `gae_lambda`, `task_reward_scale`, and
  `style_reward_scale`.
- Legacy keys such as `amp_state_preprocessor`, `*_reward_weight`, `discriminator_reward_scale`, `lambda`, and
  `clip_predicted_values` are rejected with migration hints.
- Native loading never guesses checkpoint module names. SKRL 1.x module-name conversion is explicit:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/tools/convert_legacy_checkpoint.py \
  old_agent.pt converted_agent.pt
```

The converted module names do not guarantee architecture compatibility; always validate the checkpoint in a short
play run.

## Layout

```text
source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/
├── g1_amp_env.py, g1_amp_env_cfg.py     # AMP walk/dance
├── g1_ppo_env.py                        # extensible PPO walk example
├── schema.py, skrl_compat.py             # public policy/estimator contracts
├── estimator/                           # models, adapters, collection and training
├── agents/                              # SKRL 2.x AMP/PPO YAML
├── motions/                             # reference data and conversion/view tools
├── tools/                               # replay, tracking, legacy conversion
├── train_state_estimator.py
├── train_distillation.py
├── play_with_estimator.py
└── run_ablation.py, reporting.py
```

To adapt another policy, implement the environment's `get_estimator_joint_state()` and `get_estimator_target()`
methods, define its `ObservationSchema`, and add a `PolicyAdapter`. The collection, estimator, DAgger, evaluation, and
reporting code then remains unchanged.

## License and attribution

This repository is based on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), licensed under BSD-3-Clause. See
[LICENSE](LICENSE), [LICENSE-mimic](LICENSE-mimic), and [CONTRIBUTORS.md](CONTRIBUTORS.md). The bundled G1 motion and
USD resources retain attribution to [`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp); details are
in `source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/assets/README.md`.
