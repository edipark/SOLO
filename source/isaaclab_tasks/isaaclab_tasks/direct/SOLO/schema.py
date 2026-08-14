"""Stable observation and joint schemas shared by SOLO G1 tasks and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


SCHEMA_VERSION = 1

# Canonical order exposed by the 29-DOF G1 asset. Reference NPZ files are
# name-mapped because the published walk and dance files use different orders.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_LEG_JOINT_NAMES = tuple(
    name
    for name in G1_JOINT_NAMES
    if any(token in name for token in ("hip_", "knee_", "ankle_"))
)
G1_UPPER_JOINT_NAMES = tuple(name for name in G1_JOINT_NAMES if name not in G1_LEG_JOINT_NAMES)

JOINT_PRESETS: Mapping[str, tuple[str, ...]] = {
    "all": G1_JOINT_NAMES,
    "legs": G1_LEG_JOINT_NAMES,
    "upper": G1_UPPER_JOINT_NAMES,
}

G1_KEY_BODY_NAMES = (
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "right_hip_yaw_link",
    "left_hip_yaw_link",
    "right_rubber_hand",
    "left_rubber_hand",
    "right_ankle_roll_link",
    "left_ankle_roll_link",
)


@dataclass(frozen=True)
class ObservationSchema:
    """Serializable policy/estimator interface contract."""

    name: str
    policy_dim: int
    action_dim: int = 29
    joint_position_start: int = 0
    joint_velocity_start: int = 29
    estimator_target_indices: tuple[int, ...] = ()
    estimator_target_names: tuple[str, ...] = (
        "base_lin_vel_x",
        "base_lin_vel_y",
        "base_lin_vel_z",
        "base_ang_vel_x",
        "base_ang_vel_y",
        "base_ang_vel_z",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
    )
    velocity_source: str = "sim_joint_velocity"
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.estimator_target_indices) != len(self.estimator_target_names):
            raise ValueError("Estimator target indices and names must have equal length")
        if len(set(self.estimator_target_indices)) != len(self.estimator_target_indices):
            raise ValueError("Estimator target indices must be unique")
        if self.estimator_target_indices and max(self.estimator_target_indices) >= self.policy_dim:
            raise ValueError("Estimator target index is outside the policy observation")
        if self.velocity_source != "sim_joint_velocity":
            raise ValueError("SOLO G1 only supports simulator joint velocities")

    @property
    def estimator_target_dim(self) -> int:
        return len(self.estimator_target_indices)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ObservationSchema":
        values = dict(data)
        values["estimator_target_indices"] = tuple(values.get("estimator_target_indices", ()))
        values["estimator_target_names"] = tuple(values.get("estimator_target_names", ()))
        return cls(**values)


# AMP layout: q(29), qd(29), height(1), tangent(3), normal/gravity(3),
# root linear velocity(3), root angular velocity(3), key bodies(30).
AMP_OBSERVATION_SCHEMA = ObservationSchema(
    name="g1_amp_101",
    policy_dim=101,
    estimator_target_indices=(65, 66, 67, 68, 69, 70, 62, 63, 64),
)

# PPO layout: estimated base state(9), command(3), q(29), qd(29), previous action(29).
PPO_OBSERVATION_SCHEMA = ObservationSchema(
    name="g1_ppo_walk_99",
    policy_dim=99,
    joint_position_start=12,
    joint_velocity_start=41,
    estimator_target_indices=tuple(range(9)),
)


def joint_indices(robot_joint_names: Sequence[str], preset: str = "all") -> tuple[int, ...]:
    """Resolve a preset by name, rejecting missing, duplicate, or ambiguous joints."""
    if preset not in JOINT_PRESETS:
        raise ValueError(f"Unknown joint preset {preset!r}; choose from {tuple(JOINT_PRESETS)}")
    names = tuple(robot_joint_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate robot joint names: {duplicates}")
    missing = [name for name in JOINT_PRESETS[preset] if name not in names]
    if missing:
        raise ValueError(f"Robot is missing {preset!r} estimator joints: {missing}")
    return tuple(names.index(name) for name in JOINT_PRESETS[preset])


def estimator_input_dim(preset: str = "all") -> int:
    if preset not in JOINT_PRESETS:
        raise ValueError(f"Unknown joint preset {preset!r}")
    return 2 * len(JOINT_PRESETS[preset])
