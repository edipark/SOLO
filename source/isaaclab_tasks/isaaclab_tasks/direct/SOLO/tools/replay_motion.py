"""Replay, validate, and optionally record a G1 NPZ motion in Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a SOLO G1 reference motion")
parser.add_argument("--motion", default=str(Path(__file__).parents[1] / "motions" / "G1_walk.npz"))
parser.add_argument("--record-output", default=None)
parser.add_argument("--loops", type=int, default=0, help="0 repeats until the app closes")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=600, help="Length in physics steps")
parser.add_argument("--video-dir", default=None)
parser.add_argument("--print-base-velocity", action="store_true")
parser.add_argument("--print-base-velocity-interval", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab_tasks.direct.SOLO.g1_robot_cfg import G1_SOLO_CFG
from isaaclab_tasks.direct.SOLO.motions.motion_loader import MotionLoader
from isaaclab_tasks.direct.SOLO.motions.record_data import MotionRecorder
from isaaclab_tasks.direct.SOLO.schema import G1_JOINT_NAMES


def _rgb_frame(annotator):
    raw = annotator.get_data()
    frame = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    return np.ascontiguousarray(frame[..., :3])


def main():
    if args_cli.speed <= 0.0:
        raise ValueError("--speed must be positive")
    motion = MotionLoader(args_cli.motion, args_cli.device, expected_dof_names=G1_JOINT_NAMES)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, render_interval=1)
    )
    sim.set_camera_view((3.0, 3.0, 2.0), (0.0, 0.0, 0.8))
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_SOLO_CFG.replace(prim_path="/World/Robot")
    scene = InteractiveScene(scene_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max", static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim.reset()
    robot = scene["robot"]
    dof_ids = motion.get_dof_index(robot.joint_names)
    root_id = motion.get_body_index(["pelvis"])[0]
    env_ids = torch.tensor([0], device=args_cli.device)
    recorder = MotionRecorder(
        robot,
        robot.joint_names,
        int(round(1.0 / sim.get_physics_dt())),
        args_cli.device,
        smoothing_window=1,
    )
    if args_cli.record_output:
        recorder.start_recording()
    video_writer = annotator = render_product = replicator = None
    if args_cli.video:
        import imageio.v2 as imageio
        import omni.replicator.core as rep

        replicator = rep
        video_dir = Path(args_cli.video_dir or Path(args_cli.motion).resolve().parent / "videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_dir / "g1_motion_replay.mp4", fps=30)
        render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach([render_product])

    loop = step = 0
    current_time = 0.0
    capture_every = max(1, round(1.0 / (sim.get_physics_dt() * 30.0)))
    try:
        while simulation_app.is_running() and (args_cli.loops == 0 or loop < args_cli.loops):
            if args_cli.video and step >= args_cli.video_length:
                break
            sample = motion.sample(1, np.asarray([current_time]))
            dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = sample
            root = torch.cat(
                (
                    body_pos[:, root_id],
                    body_rot[:, root_id],
                    body_lin[:, root_id] * args_cli.speed,
                    body_ang[:, root_id] * args_cli.speed,
                ),
                dim=-1,
            )
            robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
            robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(
                dof_pos[:, dof_ids], dof_vel[:, dof_ids] * args_cli.speed, None, env_ids,
            )
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
            recorder.record_frame(step)
            if video_writer is not None and step % capture_every == 0:
                frame = _rgb_frame(annotator)
                if frame is not None:
                    video_writer.append_data(frame)
            if args_cli.print_base_velocity and step % max(1, args_cli.print_base_velocity_interval) == 0:
                linear = robot.data.root_lin_vel_w[0]
                angular = robot.data.root_ang_vel_w[0]
                print(
                    f"[replay] t={current_time:.3f}s "
                    f"linear=({float(linear[0]):+.4f}, {float(linear[1]):+.4f}, {float(linear[2]):+.4f}) "
                    f"angular=({float(angular[0]):+.4f}, {float(angular[1]):+.4f}, {float(angular[2]):+.4f})"
                )
            step += 1
            current_time += sim.get_physics_dt() * args_cli.speed
            if current_time >= motion.duration:
                current_time %= motion.duration
                loop += 1
                if args_cli.record_output:
                    break
    finally:
        if args_cli.record_output:
            recorder.stop_recording()
            recorder.save_data(args_cli.record_output)
        if annotator is not None:
            annotator.detach([render_product])
        if replicator is not None and render_product is not None:
            replicator.destroy.render_product(render_product)
        if video_writer is not None:
            video_writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
