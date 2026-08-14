"""Replay, validate, and optionally record a G1 NPZ motion in Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a SOLO G1 reference motion")
parser.add_argument("--motion", default=str(Path(__file__).parents[1] / "motions" / "G1_walk.npz"))
parser.add_argument("--record-output", default=None)
parser.add_argument("--loops", type=int, default=0, help="0 repeats until the app closes")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab_tasks.direct.SOLO.g1_robot_cfg import G1_SOLO_CFG
from isaaclab_tasks.direct.SOLO.motions.motion_loader import MotionLoader
from isaaclab_tasks.direct.SOLO.motions.record_data import MotionRecorder
from isaaclab_tasks.direct.SOLO.schema import G1_JOINT_NAMES


def main():
    motion = MotionLoader(args_cli.motion, args_cli.device, expected_dof_names=G1_JOINT_NAMES)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=motion.dt, device=args_cli.device, render_interval=1))
    sim.set_camera_view((3.0, 3.0, 2.0), (0.0, 0.0, 0.8))
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_SOLO_CFG.replace(prim_path="/World/Robot")
    scene = InteractiveScene(scene_cfg)
    ground_cfg = sim_utils.CuboidCfg(
        size=(200.0, 200.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)
    sim.reset()
    robot = scene["robot"]
    dof_ids = motion.get_dof_index(robot.joint_names)
    root_id = motion.get_body_index(["pelvis"])[0]
    env_ids = torch.tensor([0], device=args_cli.device)
    recorder = MotionRecorder(robot, robot.joint_names, int(round(1.0 / motion.dt)), args_cli.device, smoothing_window=1)
    if args_cli.record_output:
        recorder.start_recording()
    loop = 0
    while simulation_app.is_running() and (args_cli.loops == 0 or loop < args_cli.loops):
        for frame in range(motion.num_frames):
            if not simulation_app.is_running():
                break
            root = torch.cat(
                (
                    motion.body_positions[frame, root_id], motion.body_rotations[frame, root_id],
                    motion.body_linear_velocities[frame, root_id], motion.body_angular_velocities[frame, root_id],
                )
            ).unsqueeze(0)
            robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
            robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(
                motion.dof_positions[frame, dof_ids].unsqueeze(0),
                motion.dof_velocities[frame, dof_ids].unsqueeze(0), None, env_ids,
            )
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
            recorder.record_frame(loop * motion.num_frames + frame)
        loop += 1
        if args_cli.record_output:
            break
    if args_cli.record_output:
        recorder.stop_recording()
        recorder.save_data(args_cli.record_output)


if __name__ == "__main__":
    main()
    simulation_app.close()
