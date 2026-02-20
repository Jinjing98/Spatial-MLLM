#!/usr/bin/env python3
"""
Format VSIBench metrics JSON to LaTeX table rows.

Usage:
    python format_metrics_latex.py metrics_qwen3-vl.json --style 1
    python format_metrics_latex.py metrics_qwen3-vl.json --style 2 --model-name "Qwen3-VL"
    python format_metrics_latex.py results/*/metrics_*.json --style 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


# JJ : Question type mappings - easy to modify
QUESTION_TYPE_ABBR = {
    "object_counting": "ObjCnt",
    "object_abs_distance": "AbsDist",
    "object_size_estimation": "ObjSz",
    "room_size_estimation": "RoomSz",
    "object_rel_distance": "RelDist",
    "object_rel_direction_easy": "RelDir-E",
    "object_rel_direction_medium": "RelDir-M",
    "object_rel_direction_hard": "RelDir-H",
    "route_planning": "RoutePl",
    "obj_appearance_order": "ApprOrd",
}

# JJ : Column order for each style - easy to modify
STYLE_CONFIGS = {
    "style1": {
        "columns": [
            "ObjCnt", "AbsDist", "ObjSz", "RoomSz", "RelDist",
            "RelDir", "RoutePl", "ApprOrd", "all_micro"
        ],
        "reldir_mode": "weighted_avg",  # weighted_avg | direct_avg | weighted_avg/direct_avg | easy/mid/hard
        "show_mra_acc": False,
    },
    "style2": {
        "columns": [
            "ObjCnt", "AbsDist", "ObjSz", "RoomSz", "RelDist",
            "RelDir",
            "RoutePl", "ApprOrd",
            "mra_micro", "acc_micro", "all_micro"
        ],
        "reldir_mode": "easy/mid/hard",  # show easy/mid/hard in one cell separated by '/'
        "show_mra_acc": True,
    },
    "style3": {
        "columns": [
            "ObjCnt", "AbsDist", "ObjSz", "RoomSz", "RelDist",
            "RelDir", "RoutePl", "ApprOrd", "all_micro"
        ],
        "reldir_mode": "weighted_avg",
        "show_mra_acc": False,
    },
}


def load_metrics(json_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_model_name(json_path: Path) -> str:
    """
    Extract model name from JSON file path with config and dataset info.
    
    Example paths:
        results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl_arkitscenes.json
        -> qwen3-vl-8f-arkitscenes
        
        results/vsibench_efficient_sampling/qwen3-vl-16f/eval_result/metrics_qwen3-vl_scannet.json
        -> qwen3-vl-16f-scannet
    """
    # Extract from path parts: .../{model_config}/eval_result/metrics_{model}_{dataset}.json
    path_parts = json_path.parts
    
    # Get model config (e.g., "qwen3-vl-8f" from directory name)
    model_config = None
    for i, part in enumerate(path_parts):
        if 'eval_result' in path_parts[i+1:i+2]:  # Next part is eval_result
            model_config = part
            break
    
    # Get dataset from filename (e.g., "arkitscenes" from "metrics_qwen3-vl_arkitscenes.json")
    filename = json_path.stem
    dataset = None
    if filename.startswith("metrics_"):
        parts = filename.split("_")
        if len(parts) >= 3:  # metrics_model_dataset
            dataset = "_".join(parts[2:])  # Handle multi-word datasets
    
    # Construct model name
    if model_config and dataset:
        return f"{model_config}-{dataset}"
    elif model_config:
        return model_config
    elif dataset:
        # Fallback: extract base model from filename
        if filename.startswith("metrics_"):
            base_model = filename.split("_")[1]
            return f"{base_model}-{dataset}"
    else:
        # Fallback: just use base model
        if filename.startswith("metrics_"):
            return filename.split("_")[1]
    
    return "Model"


def compute_reldir_values(metrics: Dict, mode: str, multiplier: float = 1.0, precision: int = 1) -> str:
    """
    Compute RelDir values based on mode.
    
    Args:
        metrics: Metrics dictionary
        mode: One of 'weighted_avg', 'direct_avg', 'weighted_avg/direct_avg', 'easy/mid/hard'
        multiplier: Multiplier for percentage display (default: 1.0)
        precision: Decimal precision
        
    Returns:
        Formatted string for RelDir cell
    """
    VALID_MODES = ["weighted_avg", "direct_avg", "weighted_avg/direct_avg", "easy/mid/hard"]
    if mode not in VALID_MODES:
        raise NotImplementedError(
            f"reldir_mode '{mode}' is not implemented. "
            f"Valid modes are: {', '.join(VALID_MODES)}"
        )
    
    per_qtype = metrics.get("per_question_type", {})
    reldir_types = [
        "object_rel_direction_easy",
        "object_rel_direction_medium", 
        "object_rel_direction_hard"
    ]
    
    scores = []
    counts = []
    
    for qtype in reldir_types:
        if qtype in per_qtype:
            scores.append(per_qtype[qtype]["score"])
            counts.append(per_qtype[qtype]["count"])
    
    if not scores:
        return "-"
    
    if mode == "weighted_avg":
        if sum(counts) > 0:
            avg = sum(s * c for s, c in zip(scores, counts)) / sum(counts)
        else:
            avg = sum(scores) / len(scores)
        return f"{avg * multiplier:.{precision}f}"
    
    elif mode == "direct_avg":
        avg = sum(scores) / len(scores)
        return f"{avg * multiplier:.{precision}f}"
    
    elif mode == "weighted_avg/direct_avg":
        if sum(counts) > 0:
            weighted_avg = sum(s * c for s, c in zip(scores, counts)) / sum(counts)
        else:
            weighted_avg = sum(scores) / len(scores)
        direct_avg = sum(scores) / len(scores)
        return f"{weighted_avg * multiplier:.{precision}f}/{direct_avg * multiplier:.{precision}f}"
    
    elif mode == "easy/mid/hard":
        # Return easy/mid/hard scores separated by '/'
        easy_score = scores[0] * multiplier if len(scores) > 0 else 0.0
        mid_score = scores[1] * multiplier if len(scores) > 1 else 0.0
        hard_score = scores[2] * multiplier if len(scores) > 2 else 0.0
        return f"{easy_score:.{precision}f}/{mid_score:.{precision}f}/{hard_score:.{precision}f}"


def format_latex_row(metrics: Dict, model_name: str, style: str = "style1", 
                     precision: int = 1, as_percentage: bool = True) -> str:
    """
    Format metrics to LaTeX table row.
    
    Args:
        metrics: Metrics dictionary
        model_name: Model name for first column
        style: Style name (style1, style2, style3)
        precision: Decimal precision for numbers
        as_percentage: If True, multiply values by 100 (default: True)
    """
    config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["style1"])
    per_qtype = metrics.get("per_question_type", {})
    
    multiplier = 100.0 if as_percentage else 1.0
    
    # Helper to get score with formatting
    def get_score(qtype_full_name: str) -> str:
        if qtype_full_name in per_qtype:
            score = per_qtype[qtype_full_name]["score"] * multiplier
            return f"{score:.{precision}f}"
        return "-"
    
    # Build row values
    values = [model_name]
    
    for col in config["columns"]:
        if col == "ObjCnt":
            values.append(get_score("object_counting"))
        elif col == "AbsDist":
            values.append(get_score("object_abs_distance"))
        elif col == "ObjSz":
            values.append(get_score("object_size_estimation"))
        elif col == "RoomSz":
            values.append(get_score("room_size_estimation"))
        elif col == "RelDist":
            values.append(get_score("object_rel_distance"))
        elif col == "RelDir":
            # Use the reldir_mode from config to format RelDir cell
            reldir_value = compute_reldir_values(metrics, config["reldir_mode"], multiplier, precision)
            values.append(reldir_value)
        elif col == "RoutePl":
            values.append(get_score("route_planning"))
        elif col == "ApprOrd":
            values.append(get_score("obj_appearance_order"))
        elif col == "mra_micro":
            mra = metrics.get("mra", {}).get("micro", 0.0) * multiplier
            values.append(f"{mra:.{precision}f}")
        elif col == "acc_micro":
            acc = metrics.get("acc", {}).get("micro", 0.0) * multiplier
            values.append(f"{acc:.{precision}f}")
        elif col == "all_micro":
            all_micro = metrics.get("all", {}).get("micro", 0.0) * multiplier
            values.append(f"{all_micro:.{precision}f}")
        else:
            values.append("-")
    
    # Format as LaTeX row
    row = " & ".join(values) + " \\\\"
    return row


def format_latex_header(style: str = "style1") -> str:
    """Generate LaTeX table header for given style."""
    config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["style1"])
    
    # Map column names to nice display names
    display_names = {
        "ObjCnt": "ObjCnt",
        "AbsDist": "AbsDist",
        "ObjSz": "ObjSz",
        "RoomSz": "RoomSz",
        "RelDist": "RelDist",
        "RelDir": "RelDir",
        "RoutePl": "RoutePl",
        "ApprOrd": "ApprOrd",
        "mra_micro": "MRA",
        "acc_micro": "ACC",
        "all_micro": "Overall",
    }
    
    headers = ["Model"] + [display_names.get(col, col) for col in config["columns"]]
    return " & ".join(headers) + " \\\\"


def main():
    parser = argparse.ArgumentParser(description="Format VSIBench metrics to LaTeX table rows")
    parser.add_argument("metrics_files", nargs="+", help="Path(s) to metrics JSON file(s)")
    parser.add_argument("--style", type=int, choices=[1, 2, 3], default=1, 
                       help="Output style (1, 2, or 3)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name (default: extract from filename)")
    parser.add_argument("--precision", type=int, default=1,
                       help="Decimal precision (default: 1)")
    parser.add_argument("--percentage", action="store_true", default=True,
                       help="Multiply values by 100 for percentage display (default: True)")
    parser.add_argument("--no-percentage", dest="percentage", action="store_false",
                       help="Show raw decimal values (0.xx) instead of percentages")
    parser.add_argument("--header", action="store_true",
                       help="Print LaTeX table header")
    parser.add_argument("--hline", action="store_true",
                       help="Add \\hline after each row")
    
    args = parser.parse_args()
    
    style_name = f"style{args.style}"
    
    # Print header if requested
    if args.header:
        print(format_latex_header(style_name))
        print("\\hline")
    
    # Process each metrics file
    for metrics_path in args.metrics_files:
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            print(f"% Warning: {metrics_path} not found", flush=True)
            continue
        
        metrics = load_metrics(metrics_path)
        model_name = args.model_name if args.model_name else extract_model_name(metrics_path)
        
        row = format_latex_row(metrics, model_name, style_name, args.precision, args.percentage)
        print(row)
        
        if args.hline:
            print("\\hline")


if __name__ == "__main__":
    main()
