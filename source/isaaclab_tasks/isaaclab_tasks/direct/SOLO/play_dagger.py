"""Play and record a schema-v2 G1 DAgger student checkpoint."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="SOLO G1 DAgger student play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default=None, help="Override the task stored in the checkpoint")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--video-dir", default="logs/solo_g1/videos/dagger")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--diagnostic-output", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path
import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.direct.SOLO.estimator.adapters import make_policy_adapter
from isaaclab_tasks.direct.SOLO.estimator.models import DaggerStudent, RunningNormalizer
from isaaclab_tasks.direct.SOLO.schema import G1_JOINT_NAMES, SCHEMA_VERSION
from isaaclab_tasks.direct.SOLO.tools.rollout_diagnostics import RolloutDiagnostics, unwrap_env_with_robot


def main():
    checkpoint = torch.load(Path(args_cli.checkpoint).resolve(), map_location=args_cli.device, weights_only=True)
    if checkpoint.get("solo_schema_version") != SCHEMA_VERSION or checkpoint.get("kind") != "dagger_student":
        raise ValueError("Only SOLO schema-v2 DAgger student checkpoints are supported")
    if checkpoint.get("velocity_source") != "sim_joint_velocity":
        raise ValueError("DAgger checkpoint does not use G1 simulator joint velocities")
    if tuple(checkpoint.get("joint_names", ())) != G1_JOINT_NAMES:
        raise ValueError("DAgger checkpoint does not use the canonical G1 29-DOF joint order")
    task = args_cli.task or checkpoint["task"]
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    raw_env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    core = unwrap_env_with_robot(raw_env)
    if core is None:
        raise RuntimeError("Unable to find the G1 DirectRLEnv")
    if args_cli.video:
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    adapter = make_policy_adapter(checkpoint["adapter"], raw_env, checkpoint["joint_preset"])
    config = checkpoint["model_config"]
    if config["input_dim"] != 58 or config["action_dim"] != 29:
        raise ValueError("DAgger checkpoint must implement the 58D-to-29D v2 interface")
    if abs(float(checkpoint["policy_dt"]) - float(core.step_dt)) > 1.0e-9:
        raise ValueError("DAgger checkpoint policy dt does not match the requested environment")
    student = DaggerStudent(
        config["input_dim"], config["action_dim"], tuple(config["hidden_dims"])
    ).to(core.device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    observation_normalizer = RunningNormalizer(config["input_dim"], core.device)
    observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
    action_normalizer = RunningNormalizer(config["action_dim"], core.device, clip=10.0)
    action_normalizer.load_state_dict(checkpoint["action_normalizer"])
    diagnostics = RolloutDiagnostics(core, 0, max_steps=max(args_cli.steps, args_cli.video_length))
    raw_env.reset()
    step_limit = args_cli.video_length if args_cli.video else args_cli.steps
    try:
        for _ in range(step_limit):
            if not simulation_app.is_running():
                break
            started = time.monotonic()
            with torch.inference_mode():
                frame = adapter.estimator_input()
                action = action_normalizer.denormalize(student(observation_normalizer.normalize(frame)))
                raw_env.step(action)
            diagnostics.record(action)
            if args_cli.real_time:
                remaining = float(core.step_dt) - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        output = args_cli.diagnostic_output or args_cli.video_dir
        artifacts = diagnostics.save(output, float(core.step_dt), "dagger_student_diagnostics")
        if artifacts:
            print(f"Diagnostics: {artifacts}")
        raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
