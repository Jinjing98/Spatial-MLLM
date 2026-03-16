#!/usr/bin/env python3
"""Utilities for plotting VsiBench checkpoint trend curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional


def load_numeric_rows(records_file: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(records_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            model_name, ckpt_step, metrics_path = line.split("|", 2)
            try:
                step_i = int(ckpt_step)
            except ValueError:
                continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    m = json.load(mf)

                acc = m.get("acc", {}) if isinstance(m, dict) else {}
                mra = m.get("mra", {}) if isinstance(m, dict) else {}
                all_m = m.get("all", {}) if isinstance(m, dict) else {}
                rows.append(
                    {
                        "model_root": model_name,
                        "ckpt_step": step_i,
                        "metrics_path": metrics_path,
                        "acc_micro": acc.get("micro"),
                        "acc_macro": acc.get("macro"),
                        "mra_micro": mra.get("micro"),
                        "mra_macro": mra.get("macro"),
                        "all_micro": all_m.get("micro"),
                        "all_macro": all_m.get("macro"),
                    }
                )
            except Exception as e:
                print(f"[WARN] failed to parse {metrics_path}: {e}")
    return rows


def parse_plot_metrics(raw: str) -> List[str]:
    if not raw.strip():
        return ["all_micro", "all_macro"]

    tokens = []
    # Supports formats like:
    # - all:micro,all:macro,acc:micro,mra:macro
    # - {"all":"micro","acc":"micro"} (duplicates cannot be preserved in this style)
    # - [{"all":"micro"},{"all":"macro"},{"acc":"micro"}]
    for left, right in re.findall(r"([A-Za-z_]+)\s*[:=]\s*['\"]?([A-Za-z_]+)['\"]?", raw):
        tokens.append((left.lower(), right.lower()))

    if not tokens:
        parts = re.split(r"[,\s;]+", raw.strip())
        for p in parts:
            if not p:
                continue
            if ":" in p:
                left, right = p.split(":", 1)
            elif "_" in p:
                left, right = p.split("_", 1)
            else:
                continue
            tokens.append((left.lower(), right.lower()))

    out: List[str] = []
    seen = set()
    for cat, agg in tokens:
        key = f"{cat}_{agg}"
        if key in {"all_micro", "all_macro", "acc_micro", "acc_macro", "mra_micro", "mra_macro"}:
            if key not in seen:
                out.append(key)
                seen.add(key)
        else:
            print(f"[WARN] unsupported plot metric '{cat}:{agg}', skip.")
    if not out:
        print("[WARN] no valid plot metrics parsed; fallback to all_micro,all_macro")
        out = ["all_micro", "all_macro"]
    return out


def metric_value(row: Dict, metric_key: str) -> Optional[float]:
    val = row.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def write_single_model_csv(rows: List[Dict], model_name: str, nframe: int, out_prefix: str) -> str:
    model_rows = [r for r in rows if r["model_root"] == model_name]
    if not model_rows:
        return ""
    model_rows = sorted(model_rows, key=lambda x: x["ckpt_step"])
    for r in model_rows:
        r["nframe"] = nframe

    csv_path = out_prefix + ".csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_root",
                "nframe",
                "ckpt_step",
                "metrics_path",
                "acc_micro",
                "acc_macro",
                "mra_micro",
                "mra_macro",
                "all_micro",
                "all_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(model_rows)
    print(f"[Trend] CSV saved: {csv_path}")
    return csv_path


def plot_single_model(rows: List[Dict], model_name: str, nframe: int, out_prefix: str, plot_metrics: List[str]) -> None:
    model_rows = [r for r in rows if r["model_root"] == model_name]
    if not model_rows:
        print(f"[Trend] No numeric checkpoint records for {model_name}, skip.")
        return
    model_rows = sorted(model_rows, key=lambda x: x["ckpt_step"])

    write_single_model_csv(rows, model_name, nframe, out_prefix)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Trend] PNG skipped (matplotlib unavailable?): {e}")
        return

    made = 0
    for metric_key in plot_metrics:
        points = [(r["ckpt_step"], metric_value(r, metric_key)) for r in model_rows]
        points = [(x, y) for x, y in points if y is not None]
        if not points:
            print(f"[WARN] Single model {model_name}: metric '{metric_key}' not found in any record, skip figure.")
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="o", label=metric_key)
        plt.xlabel("Checkpoint Step")
        plt.ylabel("Average Reward")
        plt.title(f"VsiBench Trend: {model_name} (nframe={nframe}, metric={metric_key})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        png_path = f"{out_prefix}_{metric_key}.png"
        plt.savefig(png_path, dpi=200)
        plt.close()
        made += 1
        print(f"[Trend] Plot saved: {png_path}")
    if made == 0:
        print(f"[Trend] No single-model figures generated for {model_name}.")


def plot_combined(rows: List[Dict], nframe: int, out_prefix: str, plot_metrics: List[str]) -> None:
    if not rows:
        print("[Trend] No combined records found, skip.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Trend] Combined PNG skipped (matplotlib unavailable?): {e}")
        return

    made = 0
    for metric_key in plot_metrics:
        series = defaultdict(list)
        for r in rows:
            y = metric_value(r, metric_key)
            if y is None:
                continue
            series[r["model_root"]].append((r["ckpt_step"], y))

        if not series:
            print(f"[WARN] Combined: metric '{metric_key}' not found in any record, skip figure.")
            continue

        plt.figure(figsize=(12, 6))
        for model_name, vals in sorted(series.items()):
            vals.sort(key=lambda x: x[0])
            x = [v[0] for v in vals]
            y = [v[1] for v in vals]
            plt.plot(x, y, marker="o", label=model_name)

        plt.xlabel("Checkpoint Step")
        plt.ylabel("Average Reward")
        plt.title(f"VsiBench Combined Trend Across Models (nframe={nframe}, metric={metric_key})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        png_path = f"{out_prefix}_{metric_key}.png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, dpi=200)
        plt.close()
        made += 1
        print(f"[Trend] Combined plot saved: {png_path}")

    if made == 0:
        print("[Trend] No combined figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VsiBench checkpoint trends.")
    parser.add_argument("--mode", choices=["single", "combined"], required=True)
    parser.add_argument("--records_file", required=True)
    parser.add_argument("--nframe", type=int, required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--model_root_name", default="")
    parser.add_argument(
        "--plot_metrics",
        default="all:micro,all:macro",
        help="Metric list, e.g. 'all:micro,all:macro,acc:micro,mra:macro'",
    )
    args = parser.parse_args()

    rows = load_numeric_rows(args.records_file)
    plot_metrics = parse_plot_metrics(args.plot_metrics)

    if args.mode == "single":
        if not args.model_root_name:
            raise ValueError("--model_root_name is required for mode=single")
        plot_single_model(rows, args.model_root_name, args.nframe, args.out_prefix, plot_metrics)
    else:
        plot_combined(rows, args.nframe, args.out_prefix, plot_metrics)


if __name__ == "__main__":
    main()
