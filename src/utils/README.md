# ScanNet++ DSLR Processing & Pose-Aware Sampling

Utilities for processing ScanNet++ DSLR images with pose-aware sampling strategies and visualization.

## Quick Start

```bash
# Run with default settings (uniform sampling, 20 frames)
python process_and_sample_scannetpp.py

# FPS sampling with custom parameters
python process_and_sample_scannetpp.py \
  --scene_name 00777c41d4 \
  --pose_aware_strategy fps \
  --fps_distance_mode max_norm \
  --fps_starting_mode medoid \
  --num_samples 30

# Uniform sampling on good frames only
python process_and_sample_scannetpp.py \
  --pre_filter_strategy good \
  --pose_aware_strategy uniform \
  --num_samples 20
```

## Key Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--scene_name` | `00777c41d4` | any scene | Scene to process |
| `--pre_filter_strategy` | `all` | `all`, `good` | Include all frames or filter bad poses |
| `--pose_aware_strategy` | `uniform` | `None`, `fps`, `uniform` | Sampling strategy |
| `--num_samples` | `20` | int | Number of frames to sample |
| `--fps_distance_mode` | `max_norm` | `max_norm`, `data_driven` | Distance metric for FPS |
| `--fps_starting_mode` | `first` | `first`, `rand`, `medoid` | FPS starting point |
| `--pose_analysis_target` | `sampled` | `raw_frames`, `pre_filtered`, `sampled` | Which frames to analyze |

## Visualization Outputs

- **`sampling_quality.html`**: 3D trajectory + coverage quality (min distance per frame)
- **`pose_analysis.html`**: Distance matrices + farness time series (optional, use `--plot_pose_analysis`)

Disable visualizations with `--no_visualize_sampling` or `--no_plot_pose_analysis`.

## Module Overview

- **`process_and_sample_scannetpp.py`**: Main entry point for processing and sampling
- **`pose_fps_sampling.py`**: Farthest Point Sampling (FPS) implementation
- **`pose_distance_metrics.py`**: Pose distance computation and farness metrics
- **`visualisation.py`**: Plotly-based interactive visualizations

## Pipeline

1. **Pre-filtering**: Filter frames based on pose quality (`all` vs `good`)
2. **Sampling**: Apply pose-aware strategy (`None`, `fps`, or `uniform`)
3. **Visualization**: Generate interactive HTML plots for analysis
4. **Output**: Save selected frames with metadata to JSON

## Examples

```bash
# No sampling, analyze all pre-filtered frames
python process_and_sample_scannetpp.py --pose_aware_strategy None

# FPS with medoid start, analyze raw frames
python process_and_sample_scannetpp.py \
  --pose_aware_strategy fps \
  --fps_starting_mode medoid \
  --pose_analysis_target raw_frames

# Fast processing without visualizations
python process_and_sample_scannetpp.py \
  --no_visualize_sampling \
  --no_plot_pose_analysis
```

## Help

```bash
python process_and_sample_scannetpp.py --help
```
