"""One-factor-at-a-time G1 ablation launcher with resilient reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    from .reporting import generate_report
except ImportError:
    from reporting import generate_report


EXPERIMENTS = (
    ("LSTM_DAgger_w50_all", "LSTM", 50, "all", 10),
    ("LSTM_Initial_w50_all", "LSTM", 50, "all", 0),
    ("TCN_DAgger_w50_all", "TCN", 50, "all", 10),
    ("MLP_DAgger_all", "MLP", 1, "all", 10),
    ("LSTM_DAgger_w10_all", "LSTM", 10, "all", 10),
    ("LSTM_DAgger_w25_all", "LSTM", 25, "all", 10),
    ("LSTM_DAgger_w100_all", "LSTM", 100, "all", 10),
    ("LSTM_DAgger_w50_legs", "LSTM", 50, "legs", 10),
    ("LSTM_DAgger_w50_upper", "LSTM", 50, "upper", 10),
)

TASKS = {
    "amp_walk": ("Isaac-G1-AMP-Walk-SOLO-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_dance": ("Isaac-G1-AMP-Dance-SOLO-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "ppo_walk": ("Isaac-G1-PPO-Walk-SOLO-Direct-v0", "ppo", "skrl_cfg_entry_point"),
}


def _extract_metrics(training_json: Path) -> dict:
    if not training_json.exists():
        return {}
    data = json.loads(training_json.read_text(encoding="utf-8"))
    source = data.get("metrics", data)
    metrics = {
        key: value for key, value in source.items()
        if isinstance(value, (int, float))
        or key in ("target_mae", "target_rmse", "target_names", "trace_target", "trace_prediction", "rounds")
    }
    rounds = data.get("rounds", [])
    if rounds:
        metrics["rounds"] = rounds
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SOLO G1 ablation matrix")
    parser.add_argument("--teacher-checkpoint", action="append", required=True, help="TASK=PATH; repeat per task")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default="logs/solo_g1/ablation")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.teacher_checkpoint)
    missing = [task for task in args.tasks if task not in checkpoints]
    if missing:
        raise ValueError(f"Missing --teacher-checkpoint TASK=PATH for: {missing}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.jsonl"
    matrix = []
    for task_key in args.tasks:
        for seed in args.seeds:
            matrix.append((task_key, seed, "TeacherGT", "TEACHER", 1, "all", 0))
            matrix.extend((task_key, seed, *experiment) for experiment in EXPERIMENTS)
            matrix.append((task_key, seed, "StudentDAgger_all", "STUDENT", 1, "all", 300))
    if args.fast:
        args.collect_steps, args.epochs = 100, 2
        matrix = matrix[: min(len(matrix), 6)]
    print(f"SOLO G1 ablation: {len(matrix)} runs")
    print(f"  tasks={args.tasks} seeds={args.seeds} collect={args.collect_steps} epochs={args.epochs}")
    start_all = time.monotonic()
    records = []
    baselines = {}
    for index, (task_key, seed, name, model, window, preset, dagger) in enumerate(matrix, 1):
        task, adapter, agent_key = TASKS[task_key]
        run_output = output / "runs"
        if model == "TEACHER":
            script = Path(__file__).with_name("evaluate_teacher.py")
            command = [sys.executable, str(script), "--teacher-checkpoint", checkpoints[task_key], "--task", task, "--agent", agent_key, "--adapter", adapter]
        elif model == "STUDENT":
            script = Path(__file__).with_name("train_dagger.py")
            student_output = run_output / f"{task}_StudentDAgger_seed{seed}"
            command = [
                sys.executable,
                str(script),
                "--teacher-checkpoint",
                checkpoints[task_key],
                "--task",
                task,
                "--agent",
                agent_key,
                "--adapter",
                adapter,
                "--num-iterations",
                str(dagger),
                "--rollout-steps",
                str(args.collect_steps),
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(seed),
                "--log-dir",
                str(student_output),
            ]
        else:
            script = Path(__file__).with_name("train_state_estimator.py")
            command = [sys.executable, str(script), "--teacher-checkpoint", checkpoints[task_key], "--task", task, "--agent", agent_key, "--adapter", adapter, "--estimator", model, "--window", str(window), "--joint-preset", preset, "--dagger-rounds", str(dagger)]
        if model != "STUDENT":
            command += [
                "--seed",
                str(seed),
                "--num-envs",
                str(args.num_envs),
                "--collect-steps",
                str(args.collect_steps),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                str(run_output),
            ]
        if args.headless:
            command.append("--headless")
        elapsed = time.monotonic() - start_all
        eta = elapsed / (index - 1) * (len(matrix) - index + 1) if index > 1 else 0.0
        print(f"[{index:03d}/{len(matrix):03d}] {task_key}/{name}/seed{seed} ETA={eta / 60:.1f}m")
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        started = time.monotonic()
        process = subprocess.run(command, text=True, capture_output=True)
        process_log_dir = output / "process_logs"
        process_log_dir.mkdir(parents=True, exist_ok=True)
        process_log = process_log_dir / f"{task_key}_{name}_seed{seed}.log"
        process_log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
        record = {"task": task_key, "task_id": task, "experiment": name, "seed": seed, "duration_s": time.monotonic() - started, "status": "ok" if process.returncode == 0 else "failed"}
        record["process_log"] = str(process_log)
        if process.returncode:
            record["error"] = process.stderr[-4000:] or process.stdout[-4000:]
            print(f"  FAILED ({process.returncode}); continuing")
        else:
            candidates = sorted(run_output.rglob(f"*seed{seed}/training.json"), key=lambda path: path.stat().st_mtime)
            artifact = candidates[-1] if candidates else None
            record["artifact"] = str(artifact) if artifact else None
            record["metrics"] = _extract_metrics(artifact) if artifact else {}
            if name == "TeacherGT":
                baselines[(task_key, seed)] = record["metrics"]
            baseline = baselines.get((task_key, seed), {})
            if "return_mean" in baseline and "return_mean" in record["metrics"]:
                record["metrics"]["return_delta_vs_teacher"] = record["metrics"]["return_mean"] - baseline["return_mean"]
            if record["metrics"]:
                print("  " + " ".join(f"{key}={value:.4g}" for key, value in record["metrics"].items() if isinstance(value, (int, float))))
                successful_so_far = [successful for successful in records if successful["status"] == "ok"] + [record]
                current_best = max(
                    successful_so_far,
                    key=lambda row: row.get("metrics", {}).get("return_mean", float("-inf")),
                )
                print(f"  current_best={current_best['task']}/{current_best['experiment']}/seed{current_best['seed']}")
        records.append(record)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    if not args.dry_run:
        result = generate_report(raw_path, output)
        successful = [record for record in records if record["status"] == "ok"]
        best = max(successful, key=lambda row: row.get("metrics", {}).get("return_mean", float("-inf")), default=None)
        print(f"Completed in {(time.monotonic() - start_all) / 60:.1f}m; report={result['output']}/report.md")
        if best:
            print(f"Best return: {best['task']}/{best['experiment']} seed={best['seed']} ({best.get('metrics', {}).get('return_mean', 'n/a')})")
        print("Leaderboard:")
        leaderboard = sorted(
            successful,
            key=lambda row: row.get("metrics", {}).get("return_mean", float("-inf")),
            reverse=True,
        )
        for rank, row in enumerate(leaderboard[:10], 1):
            print(
                f"  {rank:2d}. {row['task']}/{row['experiment']}/seed{row['seed']} "
                f"return={row.get('metrics', {}).get('return_mean', 'n/a')}"
            )


if __name__ == "__main__":
    main()
