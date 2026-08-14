"""29-DOF Unitree G1 AMP environment with walk/dance task rewards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .g1_amp_env_cfg import G1AmpEnvCfg
from .motions.motion_loader import MotionLoader
from .schema import G1_JOINT_NAMES, G1_KEY_BODY_NAMES
from .task_math import compose_task_reward, normalized_saturation_huber, reference_history_times


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (upper + lower)
        self.action_scale = 0.5 * (upper - lower)
        self.actions = torch.zeros((self.num_envs, 29), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        self._motion_loader = MotionLoader(self.cfg.motion_file, self.device, expected_dof_names=G1_JOINT_NAMES)
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in G1_KEY_BODY_NAMES]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(G1_KEY_BODY_NAMES)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        ground_cfg = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()

    def _apply_action(self):
        self.robot.set_joint_position_target(self.action_offset + self.action_scale * self.actions)

    def _current_amp_observation(self) -> torch.Tensor:
        return compute_amp_observation(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

    def get_estimator_target(self) -> torch.Tensor:
        obs = self._current_amp_observation()
        return torch.cat((obs[:, 65:71], obs[:, 62:65]), dim=-1)

    def get_estimator_joint_state(self) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        return self.robot.data.joint_pos, self.robot.data.joint_vel, tuple(self.robot.data.joint_names)

    def _get_observations(self) -> dict:
        obs = self._current_amp_observation()
        self.amp_observation_buffer[:, 1:] = self.amp_observation_buffer[:, :-1].clone()
        self.amp_observation_buffer[:, 0] = obs
        previous_log = self.extras.get("log", {}) if isinstance(self.extras, dict) else {}
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
            "log": previous_log,
        }
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        up = quaternion_to_tangent_and_normal(root_quat)[:, 5].clamp(-1.0, 1.0)
        upright = torch.exp(-4.0 * (1.0 - up).square())
        height_reward = torch.exp(-((height - self.cfg.nominal_height) / self.cfg.height_sigma).square())
        vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]
        if self.cfg.task_kind == "walk":
            velocity = torch.exp(-self.cfg.velocity_tracking_coeff * (vx - self.cfg.target_velocity).square())
        else:
            velocity = torch.zeros_like(vx)
        action_rate = (self.actions - self.previous_actions).square().mean(dim=-1)
        effort_limits = self.robot.data.joint_effort_limits
        saturation = normalized_saturation_huber(self.robot.data.computed_torque, effort_limits)
        raw_task = compose_task_reward(
            velocity,
            upright,
            height_reward,
            action_rate,
            saturation,
            velocity_weight=self.cfg.velocity_weight,
            upright_weight=self.cfg.upright_weight,
            height_weight=self.cfg.height_weight,
            action_rate_weight=self.cfg.action_rate_penalty,
            saturation_weight=self.cfg.saturation_penalty,
        )
        saturation_fraction = (self.robot.data.computed_torque.abs() >= effort_limits - 1.0e-5).float().mean(dim=-1)
        self.extras["log"] = {
            "reward/raw_task": raw_task.mean().detach(),
            "reward/scaled_task": raw_task.mean().detach(),
            "reward/velocity_tracking": velocity.mean().detach(),
            "reward/upright": upright.mean().detach(),
            "reward/height": height_reward.mean().detach(),
            "penalty/action_rate": action_rate.mean().detach(),
            "penalty/action_rate_weighted": (self.cfg.action_rate_penalty * action_rate.mean()).detach(),
            "penalty/saturation": saturation.mean().detach(),
            "penalty/saturation_weighted": (self.cfg.saturation_penalty * saturation.mean()).detach(),
            "metric/base_vel_x": vx.mean().detach(),
            "metric/base_height": height.mean().detach(),
            "metric/joint_velocity_squared": self.robot.data.joint_vel.square().mean().detach(),
            "metric/torque_saturation_fraction": saturation_fraction.mean().detach(),
        }
        return raw_task

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.cfg.reset_strategy == "default":
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        elif self.cfg.reset_strategy.startswith("random"):
            root_state, joint_pos, joint_vel = self._sample_motion_reset(
                env_ids, start="start" in self.cfg.reset_strategy
            )
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_motion_reset(self, env_ids: torch.Tensor, start: bool = False):
        count = env_ids.shape[0]
        times = np.zeros(count) if start else self._motion_loader.sample_times(count)
        dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = self._motion_loader.sample(count, times)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_pos[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = body_rot[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_lin[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_ang[:, self.motion_ref_body_index]
        amp = self.collect_reference_motions(count, times)
        self.amp_observation_buffer[env_ids] = amp.view(count, self.cfg.num_amp_observations, -1)
        return root_state, dof_pos[:, self.motion_dof_indexes], dof_vel[:, self.motion_dof_indexes]

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None):
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        policy_dt = self.cfg.sim.dt * self.cfg.decimation
        times = reference_history_times(current_times, self.cfg.num_amp_observations, policy_dt)
        dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = self._motion_loader.sample(
            num_samples, times
        )
        obs = compute_amp_observation(
            dof_pos[:, self.motion_dof_indexes],
            dof_vel[:, self.motion_dof_indexes],
            body_pos[:, self.motion_ref_body_index],
            body_rot[:, self.motion_ref_body_index],
            body_lin[:, self.motion_ref_body_index],
            body_ang[:, self.motion_ref_body_index],
            body_pos[:, self.motion_key_body_indexes],
        )
        return obs.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(q[..., :3])
    normal = torch.zeros_like(q[..., :3])
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return torch.cat((quat_apply(q, tangent), quat_apply(q, normal)), dim=-1)


@torch.jit.script
def compute_amp_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            joint_pos,
            joint_vel,
            root_pos[:, 2:3],
            quaternion_to_tangent_and_normal(root_quat),
            root_lin_vel,
            root_ang_vel,
            (key_body_pos - root_pos.unsqueeze(1)).reshape(key_body_pos.shape[0], -1),
        ),
        dim=-1,
    )
