#!/usr/bin/env python3
"""Aggregate six selection-layer-sweep runs into one diagnostics report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selection-layer sweep run outputs.")
    parser.add_argument("--runs-root", default="outputs/runs_selection_layer_sweep")
    parser.add_argument("--out-root", default="outputs/diagnostics")
    return parser.parse_args()


def parse_layer_key(layer_text: str) -> str:
    values = [v.strip() for v in layer_text.split("|") if v.strip()]
    return "[" + ",".join(values) + "]"


def read_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_file = run_dir / "summary.csv"
    if not summary_file.exists():
        return []
    with summary_file.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def to_float(value: str) -> float:
    return float(value)


def to_bool(value: str) -> bool:
    return value.lower() == "true"


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"runs root missing: {runs_root}")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"selection_layer_sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    combined_rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        combined_rows.extend(read_summary_rows(run_dir))

    if not combined_rows:
        print("No summary rows found.")
        return 1

    combined_path = out_dir / "combined_summary.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)

    layer_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in combined_rows:
        key = (row["method"], row["shallow_layers"])
        grouped.setdefault(key, []).append(row)
    for (method, layer), rows in grouped.items():
        loss_deltas = [to_float(r["inj_loss_delta"]) for r in rows]
        sim_deltas = [to_float(r["carrier_target_sim_delta"]) for r in rows]
        layer_rows.append(
            {
                "method": method,
                "shallow_layers": parse_layer_key(layer),
                "mean_loss_delta": mean(loss_deltas),
                "median_loss_delta": median(loss_deltas),
                "mean_sim_delta": mean(sim_deltas),
                "median_sim_delta": median(sim_deltas),
                "loss_decrease_rate": mean([to_float(r["inj_loss_end"]) < to_float(r["inj_loss_start"]) for r in rows]),
                "sim_increase_rate": mean([to_float(r["carrier_target_sim_end"]) > to_float(r["carrier_target_sim_start"]) for r in rows]),
                "linf_valid_rate": mean([to_bool(r["linf_valid"]) for r in rows]),
                "pixel_valid_rate": mean([to_bool(r["pixel_range_valid"]) for r in rows]),
            }
        )

    layer_path = out_dir / "layer_comparison.csv"
    with layer_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    pair_layer_index: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in combined_rows:
        pair_id = row["pair_id"]
        layer_key = parse_layer_key(row["shallow_layers"])
        pair_layer_index.setdefault((pair_id, layer_key), {})[row["selection_mode"]] = row

    random_vs_rows: list[dict[str, object]] = []
    for (pair_id, layer_key), slots in sorted(pair_layer_index.items()):
        fas = slots.get("fas2h")
        rnd = slots.get("random_patch")
        if not fas or not rnd:
            continue
        fas_loss_delta = to_float(fas["inj_loss_delta"])
        rnd_loss_delta = to_float(rnd["inj_loss_delta"])
        fas_sim_delta = to_float(fas["carrier_target_sim_delta"])
        rnd_sim_delta = to_float(rnd["carrier_target_sim_delta"])
        random_vs_rows.append(
            {
                "pair_id": pair_id,
                "shallow_layers": layer_key,
                "fas2h_loss_delta": fas_loss_delta,
                "random_loss_delta": rnd_loss_delta,
                "fas2h_sim_delta": fas_sim_delta,
                "random_sim_delta": rnd_sim_delta,
                "fas2h_better_loss": fas_loss_delta < rnd_loss_delta,
                "fas2h_better_sim": fas_sim_delta > rnd_sim_delta,
                "fas2h_linf_valid": to_bool(fas["linf_valid"]),
                "random_linf_valid": to_bool(rnd["linf_valid"]),
                "fas2h_pixel_valid": to_bool(fas["pixel_range_valid"]),
                "random_pixel_valid": to_bool(rnd["pixel_range_valid"]),
            }
        )

    random_vs_path = out_dir / "random_vs_fas2h_comparison.csv"
    with random_vs_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(random_vs_rows[0].keys()) if random_vs_rows else [
            "pair_id",
            "shallow_layers",
            "fas2h_loss_delta",
            "random_loss_delta",
            "fas2h_sim_delta",
            "random_sim_delta",
            "fas2h_better_loss",
            "fas2h_better_sim",
            "fas2h_linf_valid",
            "random_linf_valid",
            "fas2h_pixel_valid",
            "random_pixel_valid",
        ])
        writer.writeheader()
        writer.writerows(random_vs_rows)

    report = out_dir / "selection_layer_sweep_report.md"
    fas2h_better_loss_rate = mean([bool(row["fas2h_better_loss"]) for row in random_vs_rows]) if random_vs_rows else 0.0
    fas2h_better_sim_rate = mean([bool(row["fas2h_better_sim"]) for row in random_vs_rows]) if random_vs_rows else 0.0

    best_loss = min(layer_rows, key=lambda row: row["mean_loss_delta"])
    best_sim = max(layer_rows, key=lambda row: row["mean_sim_delta"])

    lines = [
        "# FA-S2H Selection vs Random + Layer Sweep Report",
        "",
        f"- runs_root: `{runs_root}`",
        f"- total_rows: `{len(combined_rows)}`",
        "",
        "## 关键问题回答",
        f"1. FA-S2H 在 L_inj 下降上是否优于 random: `{fas2h_better_loss_rate:.2%}` 的 pair-layer 组合中更优。",
        f"2. FA-S2H 在 carrier_target_sim 上是否优于 random: `{fas2h_better_sim_rate:.2%}` 的 pair-layer 组合中更优。",
        "3. random patch 是否也能优化 L_inj: 由 `random_loss_delta` 可直接判断，若多数为负则说明也可优化。",
        "4. 若 random 接近 FA-S2H: 说明当前 objective 对 selection 可能不敏感，或 shallow proxy 容易被多种 patch 组合满足。",
        f"5. loss 下降最多的 method+layer: `{best_loss['method']} @ {best_loss['shallow_layers']}` (mean_loss_delta={best_loss['mean_loss_delta']:.6f})。",
        f"6. sim 提升最多的 method+layer: `{best_sim['method']} @ {best_sim['shallow_layers']}` (mean_sim_delta={best_sim['mean_sim_delta']:.6f})。",
        "7. [0,1,2,3] 是否太浅: 需对比 layer_comparison.csv 的 mean_sim_delta 和稳定性指标后判断。",
        "8. [2,3,4] / [3,4,5] 是否更稳定: 需对比 smooth_decrease_rate 与 sim_increase_rate。",
        "9. per-model/per-layer 拖后腿: 需回看各 pair metrics.json 的 per_model_loss_delta/per_layer_loss_delta。",
        "10. 能说明什么: surrogate-level selection/layer 对 L_inj 与 sim 的影响。",
        "11. 不能说明什么: 不能说明闭源 MLLM/VLM 攻击成功。",
        "",
        "当前实验不能说明闭源 MLLM/VLM 攻击成功，因为本轮没有做闭源评估。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "out_dir": str(out_dir),
        "combined_summary_csv": str(combined_path),
        "layer_comparison_csv": str(layer_path),
        "random_vs_fas2h_csv": str(random_vs_path),
        "report_md": str(report),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
