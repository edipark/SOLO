# SOLO G1

### State estimation from robot joint observations

SOLO G1 trains 29-DOF Unitree G1 walk and dance policies with an
[Adversarial Motion Prior (AMP)](https://xbpeng.github.io/projects/AMP/), then replaces policy-only base information
with a learned state estimator. The G1 environment and reference motions are adapted from
[`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp); the estimator/DAgger workflow follows
`SOLO_DEXTRA`.

The default estimator uses every G1 joint: 29 joint positions plus the 29 simulator joint velocities (58D). For AMP it
predicts the complete 43D privileged suffix: height, tangent/normal basis, root velocities, and ten key-body relative
positions. PPO keeps its policy-specific 9D base-state target. No finite-difference velocity, encoder quantization, or
EMA filtering is used.

## Pipeline

### Phase 1 — privileged policy

Train an AMP walk or dance teacher with the 101D G1 motion observation. Both environments combine their task reward
with the learned style reward. With SKRL 2.x, the independent default scales are
`task_reward_scale: 0.5` and `style_reward_scale: 1.0`, matching the effective Dextra reward composition.
Physics runs at 120 Hz with decimation 4 (30 Hz policy control), and the discriminator receives four policy-spaced
AMP frames (4 x 101D = 404D). Episodes last 20 seconds. The walk task uses a 0.6 m/s reference-aligned target and
Dextra-style action-rate and normalized torque-saturation penalties; dance keeps posture/height rewards without a
velocity target.

```bash
# Walk AMP
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --algorithm AMP \
  --experiment_name dextra_aligned --headless

# Resume while explicitly restoring Dextra's initial exploration std
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --algorithm AMP \
  --checkpoint <agent.pt> --reset_log_std -1.2 --headless

# Video also writes action/position/torque diagnostics next to the recording
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --algorithm AMP \
  --checkpoint <best_agent.pt> --video --video_length 600 --print-base-velocity

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
  --teacher_checkpoint <best_agent.pt> \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 \
  --estimator LSTM --window 50 --joint-preset all --dagger-rounds 10 --headless
```

The DAgger student uses all 58 joint values and a `256-256-128` MLP to predict the teacher's 29D action. It uses an
online running normalizer, a GPU replay ring buffer, beta-decayed teacher mixing, periodic pure-student evaluation,
and saves `student_latest.pt` plus `student_best_eval.pt`.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/train_dagger.py \
  --teacher-checkpoint <best_agent.pt> --num-iterations 300 --headless
```

### Phase 3 — estimated-state inference

Teacher checkpoints are loaded through SKRL's public `Runner`/`agent.load()` API. The player uses the SKRL 2.x
evaluation API and can produce an action/estimate CSV alongside a video.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/play_teacher_with_estimator.py \
  --teacher-checkpoint <best_agent.pt> --estimator-checkpoint <best_estimator.pt> \
  --task Isaac-G1-AMP-Walk-SOLO-Direct-v0 --csv-output logs/rollout.csv --video

./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/play_dagger.py \
  --checkpoint <student_best_eval.pt> --video
```

## Motion tools

The bundled `G1_walk.npz` and `G1_dance.npz` follow the 29-joint reference schema. Available tools include schema
validation, matplotlib visualization, Isaac Sim replay/recording, conversion, pelvis alignment, and reference tracking.

```bash
# Inspect schema and values
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/verify_motion.py \
  --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/G1_walk.npz

# Isaac Sim replay; --record-output writes another compatible NPZ
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/replay_motion.py \
  --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/G1_dance.npz \
  --speed 1.0 --video --video-length 600 --print-base-velocity

# Optional side-by-side skeleton view (requires a desktop session)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/replay_motion.py \
  --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/motions/G1_walk.npz --matplotlib
```

Motion-variance generators and Dextra-specific AX18/hardware/system-identification code are intentionally excluded.

## Ablation

The runner covers teacher, estimator architecture/history/joint presets, and Student DAgger. Like SOLO_DEXTRA, one
invocation takes one teacher checkpoint and one task; `amp_walk` is the default. Use `--task amp_dance` or
`--task ppo_walk` for another task. `--seeds` is a count starting at `--seed_start`.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/run_ablation.py \
  --teacher_checkpoint <walk_amp.pt> --task amp_walk \
  --seeds 3 --seed_start 42 --headless
```

Estimator collection and Student DAgger collection are independent. Estimator ablations keep the standalone defaults
(2,000 collection steps, 500k samples, 50 initial epochs, 10 epochs per DAgger round, and 10 rounds). Ablation defaults
use 100 student iterations, 250 rollout steps and 100 optimizer steps per iteration; override them with
`--student-iterations`, `--student-collect-steps`, and `--student-train-steps`. Initial teacher rollouts are cached per
teacher/task/seed: the longest requested history up to 50 frames is projected to shorter windows and joint subsets and
shared across LSTM/TCN/MLP. The memory-heavy 100-frame experiment uses a separate cache. Model-dependent DAgger rollouts
are not shared.
Runs stay sequential on a single GPU to avoid multiple Isaac Sim processes competing for VRAM, while subprocess output
is streamed live to the terminal and `process_logs/`.
Use `--experiments LSTM_DAgger_w50_all --skip-student` for a targeted estimator rerun.

Each argument/checkpoint/code combination is isolated under `ablation/sessions/<run-signature>/`. Repeated attempts have
separate artifact and TensorBoard directories; `raw_results.jsonl` remains an audit log while reports use only the latest
record for each task/seed/experiment. Dataset cache keys include the teacher checkpoint contents and estimator/environment
implementation, and an output-directory lock rejects overlapping ablation launchers.

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
- Checkpoint loaders accept only the current schema-v2 estimator and DAgger formats. Retrain older G1 checkpoints.

## Layout

```text
source/isaaclab_tasks/isaaclab_tasks/direct/SOLO/
├── g1_amp_env.py, g1_amp_env_cfg.py     # AMP walk/dance
├── g1_ppo_env.py                        # extensible PPO walk example
├── schema.py, skrl_compat.py             # public policy/estimator contracts
├── estimator/                           # models, adapters, collection and training
├── agents/                              # SKRL 2.x AMP/PPO YAML
├── motions/                             # reference data and conversion/view tools
├── tools/                               # replay, tracking, rollout diagnostics
├── train_state_estimator.py
├── train_dagger.py
├── play_teacher_with_estimator.py, play_dagger.py
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
