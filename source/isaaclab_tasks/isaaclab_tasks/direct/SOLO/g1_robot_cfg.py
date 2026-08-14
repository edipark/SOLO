"""Self-contained Unitree G1 29-DOF articulation used by SOLO."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

G1_SOLO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "g1_29dof_rev_1_0.usd"),
        # SOLO currently derives rewards from rigid-body state, not contact reports.
        # Enabling reports for all 39 G1 bodies reserves a very large PhysX GPU buffer.
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Isaac Lab's G1_29DOF motor parameters, expressed entirely as
        # PhysX-backed implicit drives for parity with SOLO_DEXTRA.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
            effort_limit_sim={".*_hip_.*": 88.0, ".*_knee_joint": 139.0},
            velocity_limit_sim={".*_hip_.*": 32.0, ".*_knee_joint": 20.0},
            stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 200.0},
            damping={".*_hip_.*": 2.5, ".*_knee_joint": 5.0},
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=20.0,
            damping={".*_ankle_pitch_joint": 0.2, ".*_ankle_roll_joint": 0.1},
            armature=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim={"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0},
            velocity_limit_sim={"waist_yaw_joint": 32.0, "waist_roll_joint": 37.0, "waist_pitch_joint": 37.0},
            stiffness=5000.0,
            damping=5.0,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=3000.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
