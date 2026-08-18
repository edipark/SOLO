"""CPU unit tests for the SOLO G1 contracts and report pipeline."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import yaml


SOLO_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "direct" / "SOLO"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_module("solo_g1_schema_test", SOLO_DIR / "schema.py")
compat = _load_module("solo_g1_compat_test", SOLO_DIR / "skrl_compat.py")
models = _load_module("solo_g1_models_test", SOLO_DIR / "estimator" / "models.py")
reporting = _load_module("solo_g1_reporting_test", SOLO_DIR / "reporting.py")
task_math = _load_module("solo_g1_task_math_test", SOLO_DIR / "task_math.py")
motion_loader = _load_module("solo_g1_motion_loader_test", SOLO_DIR / "motions" / "motion_loader.py")
rollout_diagnostics = _load_module(
    "solo_g1_rollout_diagnostics_test", SOLO_DIR / "tools" / "rollout_diagnostics.py"
)


def _load_pipeline_module():
    package = "solo_pipeline_test_package"
    estimator_package = f"{package}.estimator"
    for name in (package, estimator_package):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules[f"{package}.schema"] = schema
    sys.modules[f"{package}.skrl_compat"] = compat
    adapters_stub = types.ModuleType(f"{estimator_package}.adapters")
    adapters_stub.PolicyAdapter = object
    sys.modules[adapters_stub.__name__] = adapters_stub
    sys.modules[f"{estimator_package}.models"] = models
    return _load_module(f"{estimator_package}.pipeline", SOLO_DIR / "estimator" / "pipeline.py")


def test_joint_presets_and_dimensions():
    assert len(schema.G1_JOINT_NAMES) == 29
    assert len(schema.G1_LEG_JOINT_NAMES) == 12
    assert len(schema.G1_UPPER_JOINT_NAMES) == 17
    assert schema.estimator_input_dim("all") == 58
    assert schema.estimator_input_dim("legs") == 24
    assert schema.estimator_input_dim("upper") == 34
    shuffled = tuple(reversed(schema.G1_JOINT_NAMES))
    ids = schema.joint_indices(shuffled, "all")
    assert tuple(shuffled[index] for index in ids) == schema.G1_JOINT_NAMES


def test_rollout_dataset_bounded_append_and_history_projection():
    pipeline = _load_pipeline_module()

    def dataset(start, count):
        ids = torch.arange(start, start + count)
        return pipeline.RolloutDataset(
            histories=ids[:, None, None].expand(-1, 4, 6).clone(),
            targets=ids[:, None].clone(),
            frames=ids[:, None].expand(-1, 6).clone(),
            teacher_actions=ids[:, None].clone(),
        )

    torch.manual_seed(7)
    combined = dataset(0, 4).append(dataset(4, 6), max_size=5)
    assert combined.histories.shape == (5, 4, 6)
    assert torch.equal(combined.histories[:, 0, 0], combined.targets[:, 0])
    assert torch.equal(combined.frames[:, 0], combined.targets[:, 0])
    assert torch.equal(combined.teacher_actions[:, 0], combined.targets[:, 0])

    source = dataset(0, 4)
    assert source.project_joint_history((0, 1, 2), 4, 4, 3) is source
    projected = source.project_joint_history((0, 2), 4, 2, 3)
    assert projected.histories.shape == (4, 2, 4)
    assert projected.frames.shape == (4, 4)


def test_joint_validation_fails_early():
    with pytest.raises(ValueError, match="missing"):
        schema.joint_indices(schema.G1_JOINT_NAMES[:-1], "all")
    with pytest.raises(ValueError, match="Duplicate"):
        schema.joint_indices((*schema.G1_JOINT_NAMES, schema.G1_JOINT_NAMES[0]), "all")


def test_reference_motions_have_the_same_joint_set():
    import numpy as np

    for name in ("G1_walk.npz", "G1_dance.npz"):
        motion = np.load(SOLO_DIR / "motions" / name)
        names = motion["dof_names"].tolist()
        assert len(names) == len(set(names)) == 29
        assert set(names) == set(schema.G1_JOINT_NAMES)


def test_motion_speed_scale_updates_timing_and_velocities():
    motion_path = SOLO_DIR / "motions" / "G1_walk.npz"
    nominal = motion_loader.MotionLoader(str(motion_path), torch.device("cpu"), speed_scale=1.0)
    faster = motion_loader.MotionLoader(str(motion_path), torch.device("cpu"), speed_scale=2.0)
    assert faster.duration == pytest.approx(nominal.duration / 2.0)
    assert torch.allclose(faster.dof_velocities, nominal.dof_velocities * 2.0)
    assert torch.allclose(faster.body_linear_velocities, nominal.body_linear_velocities * 2.0)


def test_dextra_aligned_timing_and_amp_history():
    import numpy as np

    assert task_math.PHYSICS_DT == pytest.approx(1.0 / 120.0)
    assert task_math.CONTROL_DECIMATION == 4
    assert task_math.POLICY_DT == pytest.approx(1.0 / 30.0)
    assert task_math.EPISODE_LENGTH_S / task_math.POLICY_DT == pytest.approx(600)
    assert task_math.AMP_HISTORY_STEPS * schema.AMP_OBSERVATION_SCHEMA.policy_dim == 404
    times = task_math.reference_history_times(np.array([1.0])).reshape(1, -1)
    assert times.shape == (1, 4)
    assert np.diff(times[0]) == pytest.approx([-1.0 / 30.0] * 3)


def test_normalized_actions_are_clipped_before_joint_target_mapping():
    actions = torch.tensor([[-2.0, -1.0, 0.25, 1.0, 3.0]])
    clipped = task_math.clip_normalized_actions(actions)
    assert clipped.tolist() == [[-1.0, -1.0, 0.25, 1.0, 1.0]]
    assert actions.tolist() == [[-2.0, -1.0, 0.25, 1.0, 3.0]]
    with pytest.raises(ValueError, match="positive"):
        task_math.clip_normalized_actions(actions, limit=0.0)


def test_midpoint_action_mapping_uses_soft_joint_half_range():
    limits = torch.tensor([[-2.0, 4.0], [-0.5, 1.5]])
    offset, scale = task_math.soft_limit_action_parameters(limits)
    assert torch.equal(offset, torch.tensor([1.0, 0.5]))
    assert torch.equal(scale, torch.tensor([3.0, 1.0]))
    actions = torch.tensor([[-2.0, 0.25], [1.0, 3.0]])
    targets = task_math.normalized_action_to_position(actions, offset, scale)
    assert torch.equal(targets, torch.tensor([[-2.0, 0.75], [4.0, 1.5]]))


def test_skrl_action_clipping_is_delegated_to_the_environment():
    for path in (SOLO_DIR / "agents").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["models"]["policy"]["clip_actions"] is False


def test_observation_schemas_round_trip():
    assert schema.SCHEMA_VERSION == 2
    assert schema.AMP_OBSERVATION_SCHEMA.policy_dim == 101
    assert schema.AMP_OBSERVATION_SCHEMA.estimator_target_dim == 43
    assert schema.AMP_OBSERVATION_SCHEMA.estimator_target_indices == tuple(range(58, 101))
    assert len(schema.AMP_PRIVILEGED_NAMES) == 43
    assert schema.PPO_OBSERVATION_SCHEMA.policy_dim == 99
    assert schema.ObservationSchema.from_dict(schema.AMP_OBSERVATION_SCHEMA.to_dict()) == schema.AMP_OBSERVATION_SCHEMA


def test_amp_estimator_replaces_all_privileged_columns_only():
    observations = torch.zeros(2, 101)
    observations[:, :58] = 7.0
    estimate = torch.arange(86, dtype=torch.float32).reshape(2, 43)
    injected = task_math.inject_observation_estimate(
        observations, estimate, schema.AMP_OBSERVATION_SCHEMA.estimator_target_indices
    )
    assert torch.equal(injected[:, :58], observations[:, :58])
    assert torch.equal(injected[:, 58:], estimate)
    assert torch.equal(observations[:, 58:], torch.zeros(2, 43))


@pytest.mark.parametrize("model_type,window", (("LSTM", 50), ("TCN", 50), ("MLP", 1)))
def test_estimator_shapes(model_type, window):
    estimator = models.build_estimator(model_type, input_dim=58, output_dim=43)
    sample = torch.randn(3, window, 58)
    assert estimator(sample).shape == (3, 43)
    estimator.set_normalization(sample, torch.randn(3, 43))
    assert estimator.predict(sample).shape == (3, 43)


def test_default_estimator_matches_amp_schema_v2():
    estimator = models.build_estimator("LSTM")
    assert estimator.input_dim == 58
    assert estimator.output_dim == schema.AMP_OBSERVATION_SCHEMA.estimator_target_dim == 43


def test_dagger_student_normalizer_and_replay_buffer():
    student = models.DaggerStudent()
    assert student(torch.randn(4, 58)).shape == (4, 29)
    normalizer = models.RunningNormalizer(2, "cpu")
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalizer.update(values)
    restored = models.RunningNormalizer(2, "cpu")
    restored.load_state_dict(normalizer.state_dict())
    assert torch.allclose(normalizer.normalize(values), restored.normalize(values))
    replay = models.ReplayBuffer(3, 2, 1, "cpu")
    replay.add(torch.arange(10, dtype=torch.float32).reshape(5, 2), torch.arange(5, dtype=torch.float32)[:, None])
    assert replay.size == 3
    observations, actions = replay.sample(4)
    assert observations.shape == (4, 2) and actions.shape == (4, 1)


def test_dagger_beta_schedule():
    assert models.dagger_beta(1.0, 0.5, 0.2, 0) == pytest.approx(1.0)
    assert models.dagger_beta(1.0, 0.5, 0.2, 1) == pytest.approx(0.5)
    assert models.dagger_beta(1.0, 0.5, 0.2, 10) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        models.dagger_beta(1.2, 0.5, 0.2, 0)


def test_dagger_v2_checkpoint_round_trip(tmp_path):
    student = models.DaggerStudent()
    observation_normalizer = models.RunningNormalizer(58, "cpu")
    action_normalizer = models.RunningNormalizer(29, "cpu")
    observation_normalizer.update(torch.randn(8, 58))
    action_normalizer.update(torch.randn(8, 29))
    checkpoint = {
        "solo_schema_version": schema.SCHEMA_VERSION,
        "kind": "dagger_student",
        "model_config": student.config(),
        "model_state_dict": student.state_dict(),
        "observation_normalizer": observation_normalizer.state_dict(),
        "action_normalizer": action_normalizer.state_dict(),
    }
    path = tmp_path / "student.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["solo_schema_version"] == 2
    assert loaded["kind"] == "dagger_student"
    restored = models.DaggerStudent(**{
        "input_dim": loaded["model_config"]["input_dim"],
        "action_dim": loaded["model_config"]["action_dim"],
        "hidden_dims": tuple(loaded["model_config"]["hidden_dims"]),
    })
    restored.load_state_dict(loaded["model_state_dict"])
    sample = torch.randn(4, 58)
    assert torch.allclose(student(sample), restored(sample))


def test_dextra_aligned_pipeline_entry_points():
    for name in ("train_state_estimator.py", "train_dagger.py", "play_teacher_with_estimator.py", "play_dagger.py"):
        assert (SOLO_DIR / name).is_file()
    assert not (SOLO_DIR / "train_distillation.py").exists()
    assert not (SOLO_DIR / "play_with_estimator.py").exists()


def test_ablation_uses_dextra_style_cli_cache_and_separate_student_rollout(tmp_path):
    script = SOLO_DIR / "run_ablation.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--teacher_checkpoint", str(tmp_path / "teacher.pt"),
            "--dry-run", "--fast", "--seeds", "1", "--headless",
            "--output-dir", str(tmp_path / "ablation"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert "task=amp_walk" in output
    assert "student: iterations=5, collect=50, train_steps=20" in output
    assert "--teacher_checkpoint" in output
    assert "--student-collect-steps" not in output  # translated to train_dagger --rollout-steps
    cache_lines = [line for line in output.splitlines() if "--dataset-cache" in line]
    assert len(cache_lines) == 5  # all fast-matrix estimator variants share the initial rollout
    cache_paths = [line.split("--dataset-cache ", 1)[1].split()[0] for line in cache_lines]
    assert len(set(cache_paths)) == 1
    assert all("--dataset-cache-window 50" in line for line in cache_lines)
    assert all("--max_dataset_size 20000" in line for line in cache_lines)  # --fast override


def test_ablation_estimator_defaults_match_standalone_training(tmp_path):
    script = SOLO_DIR / "run_ablation.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--teacher_checkpoint", str(tmp_path / "teacher.pt"),
            "--dry-run", "--seeds", "1", "--experiments", "LSTM_DAgger_w50_all",
            "--skip-student", "--output-dir", str(tmp_path / "ablation"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = next(line for line in result.stdout.splitlines() if "train_state_estimator.py" in line)
    assert "--collect_steps 2000" in command
    assert "--max_dataset_size 500000" in command
    assert "--epochs 50" in command
    assert "--dagger_epochs 10" in command
    assert "--dagger_rounds 10" in command
    assert "--eval_episodes 200" in command


def test_ablation_shares_short_windows_but_isolates_memory_heavy_w100(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SOLO_DIR / "run_ablation.py"),
            "--teacher_checkpoint", str(tmp_path / "teacher.pt"), "--dry-run", "--seeds", "1",
            "--experiments", "LSTM_DAgger_w10_all", "LSTM_DAgger_w50_all", "LSTM_DAgger_w100_all",
            "--skip-student", "--output-dir", str(tmp_path / "ablation"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cache_lines = [line for line in result.stdout.splitlines() if "--dataset-cache" in line]
    assert len(cache_lines) == 3
    cache_paths = [line.split("--dataset-cache ", 1)[1].split()[0] for line in cache_lines]
    assert len(set(cache_paths)) == 2
    short_lines = [line for line in cache_lines if "w100_all" not in line]
    assert len({line.split("--dataset-cache ", 1)[1].split()[0] for line in short_lines}) == 1
    assert all("--dataset-cache-window 50" in line for line in short_lines)
    assert "--dataset-cache-window 100" in next(line for line in cache_lines if "w100_all" in line)


def test_ablation_session_changes_when_teacher_contents_change(tmp_path):
    teacher = tmp_path / "teacher.pt"
    output_dir = tmp_path / "ablation"

    def session_for(contents: bytes) -> str:
        teacher.write_bytes(contents)
        result = subprocess.run(
            [
                sys.executable, str(SOLO_DIR / "run_ablation.py"),
                "--teacher_checkpoint", str(teacher), "--dry-run", "--seeds", "1",
                "--experiments", "LSTM_DAgger_w50_all", "--skip-student",
                "--output-dir", str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        line = next(line for line in result.stdout.splitlines() if line.startswith("session="))
        return line.split()[0].split("=", 1)[1]

    first = session_for(b"checkpoint-a")
    second = session_for(b"checkpoint-b")
    assert first != second
    assert (output_dir / "sessions" / first).is_dir()
    assert (output_dir / "sessions" / second).is_dir()


def test_ablation_latest_records_excludes_old_sessions_and_duplicate_attempts(tmp_path):
    sys.path.insert(0, str(SOLO_DIR))
    try:
        ablation = _load_module("solo_g1_ablation_test", SOLO_DIR / "run_ablation.py")
    finally:
        sys.path.remove(str(SOLO_DIR))
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"run_signature": "old", "task": "amp_walk", "seed": 42, "experiment": "LSTM", "value": 99},
        {"run_signature": "new", "task": "amp_walk", "seed": 42, "experiment": "LSTM", "value": 1},
        {"run_signature": "new", "task": "amp_walk", "seed": 42, "experiment": "LSTM", "value": 2},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    latest = ablation._latest_records(raw, "new")
    assert len(latest) == 1
    assert latest[0]["value"] == 2


def test_rollout_diagnostics_npz_and_plot(tmp_path):
    joint_count = 3
    data = SimpleNamespace(
        joint_names=("j0", "j1", "j2"),
        joint_pos=torch.zeros(1, joint_count),
        computed_torque=torch.ones(1, joint_count),
        applied_torque=torch.full((1, joint_count), 0.5),
        joint_effort_limits=torch.full((1, joint_count), 2.0),
    )
    core = SimpleNamespace(
        action_offset=torch.zeros(joint_count),
        action_scale=torch.ones(joint_count),
        robot=SimpleNamespace(data=data),
    )
    recorder = rollout_diagnostics.RolloutDiagnostics(core)
    recorder.record(torch.tensor([[2.0, 0.0, -2.0]]))
    artifacts = recorder.save(tmp_path, 1.0 / 30.0)
    assert Path(artifacts["data"]).is_file()
    assert Path(artifacts["plot"]).is_file()
    assert len(artifacts["joint_plots"]) == joint_count
    assert all(Path(path).is_file() for path in artifacts["joint_plots"])


def test_amp_environment_and_implicit_actuator_source():
    env_source = (SOLO_DIR / "g1_amp_env_cfg.py").read_text(encoding="utf-8")
    env_impl_source = (SOLO_DIR / "g1_amp_env.py").read_text(encoding="utf-8")
    robot_source = (SOLO_DIR / "g1_robot_cfg.py").read_text(encoding="utf-8")
    assert 'reset_strategy = "default"' in env_source
    assert "vel_window_min_vx =" in env_source
    assert "vel_window_steps = 10" in env_source
    assert "upright_weight" not in env_source
    assert "height_weight" not in env_source
    assert "target_velocity = 0.6" in env_source
    assert "velocity_reward_weight = 0.5" in env_source
    assert "velocity_reward = torch.exp" in env_impl_source
    assert "env_spacing=4.0" in env_source
    assert "GroundPlaneCfg" in (SOLO_DIR / "g1_amp_env.py").read_text(encoding="utf-8")
    assert "DCMotorCfg" not in robot_source
    assert 'effort_limit_sim={".*_hip_.*": 88.0, ".*_knee_joint": 139.0}' in robot_source
    assert 'velocity_limit_sim={".*_hip_.*": 32.0, ".*_knee_joint": 20.0}' in robot_source
    assert 'damping={".*_ankle_pitch_joint": 0.2, ".*_ankle_roll_joint": 0.1}' in robot_source
    assert 'effort_limit_sim={"waist_yaw_joint": 88.0' in robot_source
    assert 'stiffness=5000.0' in robot_source
    assert 'effort_limit_sim=300.0' in robot_source
    assert 'stiffness=3000.0' in robot_source


def test_skrl_2_yaml_and_style_scale():
    obsolete = set(compat.OBSOLETE_KEYS)
    for path in (SOLO_DIR / "agents").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not obsolete.intersection(config["agent"])
        compat.validate_skrl_config(config)
        assert config["agent"]["gae_lambda"] == pytest.approx(0.95)
        assert config["agent"]["rollouts"] == 16
        assert config["agent"]["learning_epochs"] == 6
        assert config["agent"]["mini_batches"] == 2
        assert config["agent"]["learning_rate"] == pytest.approx(5.0e-5)
        assert config["agent"]["learning_rate_scheduler"] is None
        expected_entropy = 0.0 if config["agent"]["class"] == "AMP" else 0.005
        assert config["agent"]["entropy_loss_scale"] == pytest.approx(expected_entropy)
        assert config["agent"]["time_limit_bootstrap"] is True
        assert config["models"]["policy"]["min_log_std"] == pytest.approx(-3.5)
        assert config["models"]["policy"]["initial_log_std"] == pytest.approx(-1.2)
        assert config["trainer"]["timesteps"] == 80000
        if config["agent"]["class"] == "AMP":
            expected_task_scale = 0.5 if "walk" in path.name else 0.0
            assert config["agent"]["task_reward_scale"] == pytest.approx(expected_task_scale)
            assert config["agent"]["style_reward_scale"] == pytest.approx(2.0)
            assert config["agent"]["discriminator_loss_scale"] == pytest.approx(6.0)
            assert config["models"]["policy"]["network"][0]["layers"] == [512, 256]
            assert config["models"]["discriminator"]["network"][0]["layers"] == [1024, 512, 256]
    raw_task, raw_style = torch.tensor([1.5]), torch.tensor([0.25])
    task, style, total = compat.scaled_reward(raw_task, raw_style)
    assert task.item() == pytest.approx(0.0)
    assert style.item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(0.5)


def test_amp_effective_reward_tracking_preserves_raw_rollout_reward():
    pytest.importorskip("skrl")
    from skrl.agents.torch import Agent as TorchAgent

    class IdentityPreprocessor:
        def __call__(self, value, *args, **kwargs):
            return value

    class Discriminator:
        def act(self, inputs, role):
            # sigmoid(0) produces raw style reward -log(0.5).
            observations = inputs["observations"]
            return torch.zeros((observations.shape[0], 1)), {}

    class Config:
        task_reward_scale = 0.0
        style_reward_scale = 2.0

    class FakeAmp:
        discriminator = Discriminator()
        _amp_observation_preprocessor = IdentityPreprocessor()
        cfg = Config()
        write_interval = 1
        tracking_data = defaultdict(list)
        _cumulative_rewards = None
        _cumulative_timesteps = None
        _track_rewards = deque(maxlen=100)
        _track_timesteps = deque(maxlen=100)

        def __init__(self):
            self.raw_rewards = []

        def record_transition(self, **transition):
            TorchAgent.record_transition(self, **transition)
            self.raw_rewards.append(transition["rewards"].clone())

        def track_data(self, tag, value):
            self.tracking_data[tag].append(value)

    agent = FakeAmp()
    assert compat.install_amp_reward_tracking(agent)
    assert compat.install_amp_reward_tracking(agent)

    transition = {
        "observations": torch.zeros((1, 1)),
        "states": None,
        "actions": torch.zeros((1, 1)),
        "rewards": torch.zeros((1, 1)),
        "next_observations": torch.zeros((1, 1)),
        "next_states": None,
        "terminated": torch.ones((1, 1), dtype=torch.bool),
        "truncated": torch.zeros((1, 1), dtype=torch.bool),
        "infos": {"amp_obs": torch.zeros((1, 4))},
        "timestep": 0,
        "timesteps": 1,
    }
    agent.record_transition(**transition)

    assert agent.raw_rewards[0].item() == pytest.approx(0.0)
    expected = 2.0 * -torch.log(torch.tensor(0.5)).item()
    assert agent.tracking_data["Reward / Total reward (mean)"][-1] == pytest.approx(expected)
    assert agent.tracking_data["Reward / AMP effective reward (mean)"][-1] == pytest.approx(expected)


@pytest.mark.parametrize("key", tuple(compat.OBSOLETE_KEYS))
def test_obsolete_skrl_keys_have_migration_hints(key):
    with pytest.raises(ValueError, match="Obsolete"):
        compat.validate_skrl_config({"agent": {key: 1}})


def test_force_skrl_reset_reenables_nested_reset_once_wrappers():
    inner = SimpleNamespace(_reset_once=False, _env=None)
    outer = SimpleNamespace(_reset_once=False, _env=inner)
    compat.force_skrl_isaaclab_reset(outer)
    assert outer._reset_once is True
    assert inner._reset_once is True


def test_report_artifacts_and_failed_run(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "task": "amp_walk",
            "experiment": "LSTM",
            "seed": 42,
            "status": "ok",
            "metrics": {
                "rmse": 0.2,
                "r2": 0.8,
                "return_mean": 10.0,
                "episode_length_mean": 580.0,
                "episode_length_std": 25.0,
                "death_rate": 3.0,
                "timeout_rate": 97.0,
                "target_rmse": [0.1] * 9,
                "trace_target": [[0.0] * 9, [1.0] * 9],
                "trace_prediction": [[0.1] * 9, [0.9] * 9],
                "rounds": [
                    {"round": 0, "training": {"best_validation_mse": 0.4}},
                    {"round": 1, "training": {"best_validation_mse": 0.2}},
                ],
            },
        },
        {
            "task": "amp_walk", "experiment": "LSTM", "seed": 43, "status": "ok",
            "metrics": {
                "rmse": 0.3, "r2": 0.7, "return_mean": 12.0,
                "episode_length_mean": 600.0, "episode_length_std": 0.0,
                "death_rate": 0.0, "timeout_rate": 100.0,
            },
        },
        {"task": "amp_walk", "experiment": "TCN", "seed": 42, "status": "failed", "error": "synthetic failure"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = reporting.generate_report(raw, tmp_path / "report")
    assert result["runs"] == 3 and result["failures"] == 1
    for name in ("summary.json", "summary.csv", "results_tidy.csv", "table.md", "table.tex", "report.md"):
        assert (tmp_path / "report" / name).is_file()
    assert "synthetic failure" in (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    table = (tmp_path / "report" / "table.md").read_text(encoding="utf-8")
    assert "Episode steps" in table and "590.0 ± 14.1" in table
    assert "Death %" in table and "Timeout %" in table
    if result["plots"]:
        for name in (
            "episode_length_mean.png", "timeout_rate.png", "target_rmse_heatmap.png",
            "dagger_learning_curve.png", "representative_trace.png",
        ):
            assert (tmp_path / "report" / name).is_file()
