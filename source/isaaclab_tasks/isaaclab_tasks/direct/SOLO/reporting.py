"""Ablation aggregation, tables, plots, and human-readable report generation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


PRIMARY_METRICS = (
    "rmse", "mae", "r2", "return_mean", "episode_length_mean", "death_rate", "timeout_rate",
    "teacher_action_mse", "action_smoothness", "energy", "torque_rms", "inference_ms_per_sample", "parameters",
    "success_rate", "base_linear_speed", "base_angular_speed", "action_saturation", "torque_saturation",
    "raw_task_reward", "amp_raw_style", "amp_scaled_task", "amp_scaled_style", "amp_effective_reward",
)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric(values: Iterable) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("task", "unknown"), row.get("experiment", "unknown"))].append(row)
    output = []
    for (task, experiment), group in sorted(groups.items()):
        summary = {"task": task, "experiment": experiment, "seeds": len(group), "successful": sum(row.get("status") == "ok" for row in group)}
        for metric in PRIMARY_METRICS:
            values = _numeric(row.get("metrics", {}).get(metric) for row in group)
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / math.sqrt(len(values))
        output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict]) -> str:
    columns = ["task", "experiment", "successful", "rmse_mean", "r2_mean", "return_mean_mean", "death_rate_mean", "inference_ms_per_sample_mean"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(f"{row.get(column, ''):.5g}" if isinstance(row.get(column), float) else str(row.get(column, "")) for column in columns) + " |")
    return "\n".join((header, divider, *body))


def _latex_table(rows: list[dict]) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Task & Experiment & RMSE & $R^2$ & Return & Death (\%) \\", r"\midrule"]
    for row in rows:
        lines.append(
            f"{row['task']} & {row['experiment']} & {row.get('rmse_mean', float('nan')):.4f} & "
            f"{row.get('r2_mean', float('nan')):.4f} & {row.get('return_mean_mean', float('nan')):.2f} & "
            f"{row.get('death_rate_mean', float('nan')):.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _plots(output: Path, rows: list[dict], raw_rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    labels = [f"{row['task'].replace('Isaac-G1-', '')}\n{row['experiment']}" for row in rows]
    for metric, title in (("rmse", "Estimator RMSE"), ("return_mean", "Closed-loop return"), ("death_rate", "Death rate")):
        selected = [(index, row) for index, row in enumerate(rows) if f"{metric}_mean" in row]
        if not selected:
            continue
        x = range(len(selected))
        values = [row[f"{metric}_mean"] for _, row in selected]
        errors = [row.get(f"{metric}_ci95", 0.0) for _, row in selected]
        figure, axis = plt.subplots(figsize=(max(8, len(selected) * 0.8), 5))
        axis.errorbar(x, values, yerr=errors, fmt="o", capsize=4)
        axis.set_xticks(list(x), [labels[index] for index, _ in selected], rotation=35, ha="right")
        axis.set_title(f"{title} (mean and 95% CI)")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{metric}.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    # Performance-latency Pareto plot.
    selected = [row for row in rows if "rmse_mean" in row and "inference_ms_per_sample_mean" in row]
    if selected:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter([row["inference_ms_per_sample_mean"] for row in selected], [row["rmse_mean"] for row in selected])
        for row in selected:
            axis.annotate(row["experiment"], (row["inference_ms_per_sample_mean"], row["rmse_mean"]), fontsize=7)
        axis.set(xlabel="Inference latency (ms/sample)", ylabel="RMSE", title="Accuracy/latency Pareto")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"pareto.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    targets = [
        (f"{row.get('experiment')}/s{row.get('seed')}", row.get("metrics", {}).get("target_rmse"))
        for row in raw_rows if row.get("status") == "ok" and row.get("metrics", {}).get("target_rmse")
    ]
    if targets:
        figure, axis = plt.subplots(figsize=(10, max(4, len(targets) * 0.35)))
        image = axis.imshow([values for _, values in targets], aspect="auto")
        axis.set_yticks(range(len(targets)), [label for label, _ in targets], fontsize=7)
        axis.set_xticks(range(9), ["lvx", "lvy", "lvz", "avx", "avy", "avz", "gx", "gy", "gz"])
        axis.set_title("Estimator target RMSE heatmap")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"target_rmse_heatmap.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    dagger_rows = [row for row in raw_rows if row.get("metrics", {}).get("rounds")]
    if dagger_rows:
        figure, axis = plt.subplots(figsize=(8, 5))
        for row in dagger_rows:
            rounds = row["metrics"]["rounds"]
            x = [item["round"] for item in rounds]
            y = [item.get("training", {}).get("best_validation_mse", float("nan")) for item in rounds]
            axis.plot(x, y, alpha=0.55, label=f"{row['experiment']}/s{row['seed']}")
        axis.set(xlabel="DAgger round", ylabel="Validation MSE", title="DAgger learning curves")
        if len(dagger_rows) <= 12:
            axis.legend(fontsize=6)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"dagger_learning_curve.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    trace_row = next(
        (row for row in raw_rows if row.get("metrics", {}).get("trace_target") and row.get("metrics", {}).get("trace_prediction")),
        None,
    )
    if trace_row:
        target = trace_row["metrics"]["trace_target"]
        prediction = trace_row["metrics"]["trace_prediction"]
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for group, axis in enumerate(axes):
            for offset in range(3):
                index = group * 3 + offset
                axis.plot([row[index] for row in target], alpha=0.65, label=f"target {index}")
                axis.plot([row[index] for row in prediction], linestyle="--", alpha=0.65, label=f"estimate {index}")
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, fontsize=6)
        axes[0].set_title(f"Representative estimator trace: {trace_row['experiment']}")
        axes[-1].set_xlabel("Sample")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"representative_trace.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    return artifacts


def generate_report(raw_jsonl: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(raw_jsonl)
    summary = aggregate(raw_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output / "results_tidy.csv", [
        {**{key: value for key, value in row.items() if key != "metrics"}, **row.get("metrics", {})} for row in raw_rows
    ])
    _write_csv(output / "summary.csv", summary)
    table = _markdown_table(summary)
    (output / "table.md").write_text(table + "\n", encoding="utf-8")
    (output / "table.tex").write_text(_latex_table(summary) + "\n", encoding="utf-8")
    try:
        plots = _plots(output, summary, raw_rows)
        plot_error = None
    except ImportError as exc:
        plots = []
        plot_error = str(exc)
        (output / "PLOTS_UNAVAILABLE.txt").write_text(f"Install matplotlib to generate plots: {exc}\n", encoding="utf-8")
    failures = [row for row in raw_rows if row.get("status") != "ok"]
    report = [
        "# SOLO G1 Ablation Report", "", f"Runs: {len(raw_rows)}; failures: {len(failures)}", "",
        "The default AMP reward scales are task=1.0 and style=2.0. Scales are independent and do not sum to one.", "",
        "## Results", "", table, "", "## Artifacts", "",
        f"Plots: {', '.join(plots) if plots else 'none'}",
    ]
    if plot_error:
        report.extend(("", f"Plot generation warning: {plot_error}"))
    if failures:
        report.extend(("", "## Failed runs", "", *[f"- {row.get('task')}/{row.get('experiment')}/seed{row.get('seed')}: {row.get('error')}" for row in failures]))
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"runs": len(raw_rows), "failures": len(failures), "plots": plots, "output": str(output)}
