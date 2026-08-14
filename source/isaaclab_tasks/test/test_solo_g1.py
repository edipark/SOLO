"""CPU unit tests for the SOLO G1 contracts and report pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

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


def test_observation_schemas_round_trip():
    assert schema.AMP_OBSERVATION_SCHEMA.policy_dim == 101
    assert schema.AMP_OBSERVATION_SCHEMA.estimator_target_dim == 9
    assert schema.PPO_OBSERVATION_SCHEMA.policy_dim == 99
    assert schema.ObservationSchema.from_dict(schema.AMP_OBSERVATION_SCHEMA.to_dict()) == schema.AMP_OBSERVATION_SCHEMA


@pytest.mark.parametrize("model_type,window", (("LSTM", 50), ("TCN", 50), ("MLP", 1)))
def test_estimator_shapes(model_type, window):
    estimator = models.build_estimator(model_type, input_dim=58, output_dim=9)
    sample = torch.randn(3, window, 58)
    assert estimator(sample).shape == (3, 9)
    estimator.set_normalization(sample, torch.randn(3, 9))
    assert estimator.predict(sample).shape == (3, 9)


def test_vanilla_student_shape():
    assert models.VanillaStudent()(torch.randn(4, 58)).shape == (4, 29)


def test_skrl_2_yaml_and_style_scale():
    obsolete = set(compat.OBSOLETE_KEYS)
    for path in (SOLO_DIR / "agents").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not obsolete.intersection(config["agent"])
        compat.validate_skrl_config(config)
        assert config["agent"]["gae_lambda"] == pytest.approx(0.95)
        if config["agent"]["class"] == "AMP":
            assert config["agent"]["task_reward_scale"] == pytest.approx(1.0)
            assert config["agent"]["style_reward_scale"] == pytest.approx(2.0)
    raw_task, raw_style = torch.tensor([1.5]), torch.tensor([0.25])
    task, style, total = compat.scaled_reward(raw_task, raw_style)
    assert task.item() == pytest.approx(1.5)
    assert style.item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(2.0)


@pytest.mark.parametrize("key", tuple(compat.OBSOLETE_KEYS))
def test_obsolete_skrl_keys_have_migration_hints(key):
    with pytest.raises(ValueError, match="Obsolete"):
        compat.validate_skrl_config({"agent": {key: 1}})


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
                "target_rmse": [0.1] * 9,
                "trace_target": [[0.0] * 9, [1.0] * 9],
                "trace_prediction": [[0.1] * 9, [0.9] * 9],
                "rounds": [
                    {"round": 0, "training": {"best_validation_mse": 0.4}},
                    {"round": 1, "training": {"best_validation_mse": 0.2}},
                ],
            },
        },
        {"task": "amp_walk", "experiment": "LSTM", "seed": 43, "status": "ok", "metrics": {"rmse": 0.3, "r2": 0.7, "return_mean": 12.0}},
        {"task": "amp_walk", "experiment": "TCN", "seed": 42, "status": "failed", "error": "synthetic failure"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = reporting.generate_report(raw, tmp_path / "report")
    assert result["runs"] == 3 and result["failures"] == 1
    for name in ("summary.json", "summary.csv", "results_tidy.csv", "table.md", "table.tex", "report.md"):
        assert (tmp_path / "report" / name).is_file()
    assert "synthetic failure" in (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    if result["plots"]:
        for name in ("target_rmse_heatmap.png", "dagger_learning_curve.png", "representative_trace.png"):
            assert (tmp_path / "report" / name).is_file()
