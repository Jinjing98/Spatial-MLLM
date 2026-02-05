import json
import os
from pathlib import Path


def load_metrics(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_model_name(file_path, results_root):
    """
    Extract model name from file path relative to results root
    
    Args:
        file_path: Full path to the metrics JSON file
        results_root: Root directory for results
    
    Returns:
        Relative path from results_root to the model directory
    """
    file_path = Path(file_path).resolve()
    results_root = Path(results_root).resolve()
    
    # Get relative path from results_root
    try:
        rel_path = file_path.relative_to(results_root)
        # Remove the 'eval_result/metrics_*.json' part to get model directory
        # e.g., vsibench/data-16f_scannet/eval_result/metrics_spatial-mllm.json
        # -> vsibench/data-16f_scannet
        parts = rel_path.parts[:-2]  # Remove eval_result and filename
        return '/'.join(parts) if parts else rel_path.stem.replace('metrics_', '')
    except ValueError:
        # If file_path is not relative to results_root, fall back to filename
        return file_path.stem.replace('metrics_', '')


def format_value(value, baseline_value=None, compute_change_ratio=False, is_best=False):
    """
    Format a value with optional change ratio and bold highlighting
    
    Args:
        value: The metric value to format
        baseline_value: The baseline value for comparison (if computing change ratio)
        compute_change_ratio: Whether to compute and display change ratio
        is_best: Whether this is the best value (for bold highlighting)
    
    Returns:
        Formatted string with value and optional change ratio
    """
    value_str = f"{value:.4f}"
    
    if is_best:
        value_str = f"**{value_str}**"
    
    change_str = ""
    if compute_change_ratio and baseline_value is not None and value != baseline_value:
        change = ((value - baseline_value) / baseline_value) * 100
        change_str = f"({change:+.2f}%)"
    
    return value_str, change_str


def compare_metrics(baseline_path, compared_paths, results_root, highlight_best_flag=True, compute_change_ratio_flag=True):
    """
    Compare metrics across multiple models and print formatted table
    
    Args:
        baseline_path: Path to baseline metrics JSON
        compared_paths: List of paths to compared model metrics JSONs
        results_root: Root directory for results (used to compute relative paths for model names)
        highlight_best_flag: Whether to highlight best values with bold
        compute_change_ratio_flag: Whether to compute change ratios relative to baseline
    """
    # Load all metrics
    all_paths = [baseline_path] + compared_paths
    all_metrics = [load_metrics(path) for path in all_paths]
    model_names = [get_model_name(path, results_root) for path in all_paths]
    
    # Get all question types (union of all models)
    question_types_set = set()
    for metrics in all_metrics:
        question_types_set.update(metrics['per_question_type'].keys())
    question_types = sorted(list(question_types_set))
    
    # Print table header
    print("\n" + "="*100)
    print("METRICS COMPARISON TABLE")
    print("="*100 + "\n")
    
    # Build column headers
    col_widths = [max(30, max(len(qt) for qt in question_types))]  # First column for question type
    for name in model_names:
        col_widths.append(max(15, len(name) + 2))
    
    # Print header row
    header = f"{'Question Type (Count)':<{col_widths[0]}}"
    for i, name in enumerate(model_names):
        header += f" | {name:^{col_widths[i+1]}}"
    print(header)
    print("-" * len(header))
    
    # Print per-question-type metrics
    for qt in question_types:
        # Get counts and scores for this question type across all models
        # Use None if the question type doesn't exist for a model
        counts = []
        scores = []
        for metrics in all_metrics:
            if qt in metrics['per_question_type']:
                counts.append(metrics['per_question_type'][qt]['count'])
                scores.append(metrics['per_question_type'][qt]['score'])
            else:
                counts.append(None)
                scores.append(None)
        
        # Find best score index (only among valid scores)
        valid_scores = [s for s in scores if s is not None]
        if len(valid_scores) > 0 and highlight_best_flag:
            max_score = max(valid_scores)
            best_idx = scores.index(max_score)
        else:
            best_idx = -1
        
        # First row: question name and values
        row1 = f"{qt:<{col_widths[0]}}"
        value_strs = []
        change_strs = []
        for i, score in enumerate(scores):
            if score is None:
                value_strs.append('-')
                change_strs.append('-')
            else:
                baseline_score = scores[0] if compute_change_ratio_flag and i > 0 and scores[0] is not None else None
                value_str, change_str = format_value(
                    score, 
                    baseline_score, 
                    compute_change_ratio_flag and i > 0,
                    is_best=(i == best_idx and highlight_best_flag)
                )
                value_strs.append(value_str)
                change_strs.append(change_str if change_str else '-')
        
        for i, value_str in enumerate(value_strs):
            row1 += f" | {value_str:^{col_widths[i+1]}}"
        print(row1)
        
        # Second row: counts and change ratios
        count_displays = [str(c) if c is not None else '-' for c in counts]
        count_str = f"({'/'.join(count_displays)})"
        row2 = f"{count_str:<{col_widths[0]}}"
        for i, change_str in enumerate(change_strs):
            row2 += f" | {change_str:^{col_widths[i+1]}}"
        print(row2)
        print()
    
    # Print separator before overall metrics
    print("="*len(header))
    print("OVERALL METRICS")
    print("="*len(header) + "\n")
    
    # Calculate total counts for each model
    total_counts = []
    for metrics in all_metrics:
        total = sum(qt_data['count'] for qt_data in metrics['per_question_type'].values())
        total_counts.append(total)
    
    # Print overall metrics (acc, mra, all) with micro and macro
    for metric_name in ['acc', 'mra', 'all']:
        for sub_metric in ['micro', 'macro']:
            # Get values
            values = [metrics[metric_name][sub_metric] for metrics in all_metrics]
            best_idx = values.index(max(values)) if highlight_best_flag else -1
            
            # First row: metric name and values
            row_name = f"{metric_name}_{sub_metric}"
            row1 = f"{row_name:<{col_widths[0]}}"
            
            value_strs = []
            change_strs = []
            for i, value in enumerate(values):
                baseline_value = values[0] if compute_change_ratio_flag and i > 0 else None
                value_str, change_str = format_value(
                    value,
                    baseline_value,
                    compute_change_ratio_flag and i > 0,
                    is_best=(i == best_idx and highlight_best_flag)
                )
                value_strs.append(value_str)
                change_strs.append(change_str if change_str else '-')
            
            for i, value_str in enumerate(value_strs):
                row1 += f" | {value_str:^{col_widths[i+1]}}"
            print(row1)
            
            # Second row: total counts and change ratios
            count_str = f"({'/'.join(map(str, total_counts))})"
            row2 = f"{count_str:<{col_widths[0]}}"
            for i, change_str in enumerate(change_strs):
                row2 += f" | {change_str:^{col_widths[i+1]}}"
            print(row2)
            print()
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    # Configuration
    RESULTS_ROOT = "/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/results"
    
    # Model paths (can be absolute or relative to RESULTS_ROOT)
    # None SA
    baseline_path = f"{RESULTS_ROOT}/vsibench/suspicious_data-16f-all3datasets/eval_result/metrics_spatial-mllm.json"
    compared_paths = [
        f"{RESULTS_ROOT}/vsibench/custom-spatial-mllmmrope_Pose-custom-spatial-mllm-16f_arkitscenes/eval_result/metrics_custom-spatial-mllm.json",
    ]

    # SA
    baseline_path = f"{RESULTS_ROOT}/vsibench-sa-sampling/spatial-mllm-16f_all/eval_result/metrics_spatial-mllm.json"
    compared_paths = [
        f"{RESULTS_ROOT}/vsibench-sa-sampling/spatial-mllm-spatial-mllm-16f_all/eval_result/metrics_spatial-mllm.json",
    ]

    

    
    # Options
    highlight_best_flag = True
    compute_change_ratio_flag = True
    
    compare_metrics(baseline_path, compared_paths, RESULTS_ROOT, highlight_best_flag, compute_change_ratio_flag)
