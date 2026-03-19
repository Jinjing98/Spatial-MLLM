#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch


def _resolve_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y")


def _load_video_records(dataset_use: str):
    """jj: Build video list with the same dataset parsing/sampling logic as training data init."""
    from src.qwenvl.data import data_list
    from src.qwenvl.data.data_qwen import read_jsonl

    dataset_names = [x.strip() for x in dataset_use.split(",") if x.strip()]
    dataset_cfgs = data_list(dataset_names)

    random.seed(42)
    records: Dict[str, Dict[str, Any]] = {}

    for cfg in dataset_cfgs:
        file_format = cfg["annotation_path"].split(".")[-1]
        if file_format == "jsonl":
            annotations = read_jsonl(cfg["annotation_path"])
        else:
            with open(cfg["annotation_path"], "r", encoding="utf-8") as f:
                annotations = json.load(f)

        sampling_rate = cfg.get("sampling_rate", 1.0)
        if sampling_rate < 1.0:
            annotations = random.sample(annotations, int(len(annotations) * sampling_rate))

        dataset_name = Path(cfg["annotation_path"]).stem
        for ann in annotations:
            if "video" not in ann:
                continue
            video_root = ann.get("video_root", cfg["data_path"])
            video_field = ann["video"]
            video_list = video_field if isinstance(video_field, list) else [video_field]

            for rel_video in video_list:
                if not isinstance(rel_video, str) or not rel_video:
                    continue
                abs_video = os.path.normpath(os.path.join(video_root, rel_video))
                if abs_video not in records:
                    records[abs_video] = {
                        "video_abs_path": abs_video,
                        "video_key": rel_video,  # jj: key follows annotation path string.
                        "video_root": video_root,
                        "dataset_name": dataset_name,
                    }

    out = [records[k] for k in sorted(records.keys())]
    return out


def _make_data_stub(args):
    from src.qwenvl.data.data_qwen import LazySupervisedDataset
    from src.qwenvl.preprocessor.image_processing_qwen2_vl import Qwen2VLImageProcessorModified

    processor = Qwen2VLImageProcessorModified.from_pretrained(args.pretrained_model_name_or_path)
    data_args = SimpleNamespace(
        # jj: Keep same video sampling controls as training.
        base_interval=args.base_interval,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        video_frame_fps=args.video_frame_fps,
        sampling_enforce_real_neighbour=args.sampling_enforce_real_neighbour,
        neighbour_mode=args.neighbour_mode,
        neighbour_max_step=args.neighbour_max_step,
        video_max_frame_pixels=args.video_max_frame_pixels,
        video_min_frame_pixels=args.video_min_frame_pixels,
        image_processor=processor,
    )

    # jj: Mirror dataset init processor constraints.
    data_args.image_processor.max_pixels = args.max_pixels
    data_args.image_processor.min_pixels = args.min_pixels
    data_args.image_processor.size["longest_edge"] = args.max_pixels
    data_args.image_processor.size["shortest_edge"] = args.min_pixels

    stub = LazySupervisedDataset.__new__(LazySupervisedDataset)
    stub.data_args = data_args
    return stub


def _sample_novel_indices_like_dataset(input_frame_indices: np.ndarray) -> np.ndarray:
    """jj: Reproduce data_qwen._get_nvs_target_frames sampling exactly (indices only)."""
    frame_idx = np.asarray(input_frame_indices, dtype=int)
    n_in = len(frame_idx)
    input_set = set(frame_idx.tolist())
    first_in, last_in = int(frame_idx[0]), int(frame_idx[-1])
    available = sorted(set(range(first_in + 1, last_in)) - input_set)

    if not available:
        dummy_idx = int(frame_idx[n_in // 2])
        return np.asarray([dummy_idx], dtype=int)

    n_sample = min(n_in, len(available))
    target_indices = sorted(random.sample(available, n_sample))
    return np.asarray(target_indices, dtype=int)


def _output_pt_path(output_root: Path, video_key: str) -> Path:
    norm = video_key.lstrip("/")
    p = Path(norm)
    if p.suffix:
        p = p.with_suffix(".pt")
    else:
        p = Path(str(p) + ".pt")
    return output_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute camera info (VGGT) for full training dataset.")
    parser.add_argument("--dataset_use", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vggt_checkpoints_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base_interval", type=float, default=2)# was 4 when gen the 1st version preproceed vggt cam info
    parser.add_argument("--video_min_frames", type=int, default=16)
    parser.add_argument("--video_max_frames", type=int, default=16)
    parser.add_argument("--video_frame_fps", type=float, default=4)
    parser.add_argument("--sampling_enforce_real_neighbour", type=_resolve_bool, default=True)
    parser.add_argument("--neighbour_mode", type=str, default="after", choices=["before", "after", "random"])
    parser.add_argument("--neighbour_max_step", type=int, default=1)

    parser.add_argument("--max_pixels", type=int, default=324576)
    parser.add_argument("--min_pixels", type=int, default=293216)
    parser.add_argument("--video_max_frame_pixels", type=int, default=324576)
    parser.add_argument("--video_min_frame_pixels", type=int, default=293216)

    parser.add_argument("--limit", type=int, default=0, help="0 means all.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--skip_existing", type=_resolve_bool, default=True)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards-1}]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"report_shard{args.shard_id:03d}_of_{args.num_shards:03d}.jsonl"
    summary_path = output_root / f"summary_shard{args.shard_id:03d}_of_{args.num_shards:03d}.json"

    # jj: DATASET_ROOT must be set before importing src.qwenvl.data package.
    if "DATASET_ROOT" not in os.environ:
        raise ValueError("DATASET_ROOT environment variable is required.")

    records = _load_video_records(args.dataset_use)
    if args.limit > 0:
        records = records[:args.limit]

    # jj: Deterministic shard split by sorted global index.
    records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_id]

    print("=" * 100)
    print("[INFO] Full-dataset VGGT precompute")
    print(f"[INFO] DATASET_ROOT={os.environ.get('DATASET_ROOT')}")
    print(f"[INFO] dataset_use={args.dataset_use}")
    print(f"[INFO] records_in_shard={len(records)}")
    print(f"[INFO] output_root={output_root}")
    print("=" * 100)

    data_stub = _make_data_stub(args)

    from src.qwenvl.model.spatial_encoder import VGGTSpatialEncoderConfig, VGGTSpatialEncoderPreTrainedModel
    from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")

    spatial_encoder = VGGTSpatialEncoderPreTrainedModel(VGGTSpatialEncoderConfig())
    spatial_encoder.load_pretrained_weights(args.vggt_checkpoints_path)
    spatial_encoder = spatial_encoder.to(device).eval()

    stats = {
        "total": len(records),
        "ok": 0,
        "skip_existing": 0,
        "failed": 0,
        "start_time": time.time(),
    }

    with open(report_path, "w", encoding="utf-8") as rep:
        for idx, rec in enumerate(records):
            t0 = time.time()
            video_abs_path = rec["video_abs_path"]
            video_key = rec["video_key"]
            out_pt = _output_pt_path(output_root, video_key)

            item = {
                "idx": idx,
                "video_key": video_key,
                "video_abs_path": video_abs_path,
                "output_pt": str(out_pt),
                "status": "",
                "error": None,
                "elapsed_sec": None,
            }

            try:
                if args.skip_existing and out_pt.exists():
                    item["status"] = "skip_existing"
                    stats["skip_existing"] += 1
                    item["elapsed_sec"] = round(time.time() - t0, 4)
                    rep.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                if not os.path.exists(video_abs_path):
                    raise FileNotFoundError(f"video not found: {video_abs_path}")

                _, _, _, input_video_tchw, input_frame_idx = data_stub.process_video(video_abs_path)
                input_frame_idx = np.asarray(input_frame_idx, dtype=int)
                input_frame_idx_tensor = torch.as_tensor(input_frame_idx, dtype=torch.long)

                # jj: Snapshot random state so novel_pool_indices exactly match _get_nvs_target_frames internals.
                rnd_state = random.getstate()
                nvs_target_tchw, nvs_is_input_mask = data_stub._get_nvs_target_frames(
                    video_abs_path, input_video_tchw, input_frame_indices=input_frame_idx
                )
                random.setstate(rnd_state)
                novel_pool_indices = _sample_novel_indices_like_dataset(input_frame_idx)
                novel_pool_indices_tensor = torch.as_tensor(novel_pool_indices, dtype=torch.long)

                nvs_mask = nvs_is_input_mask.to(dtype=torch.bool, device=input_video_tchw.device)
                interleaved = torch.zeros(
                    nvs_mask.shape[0],
                    *input_video_tchw.shape[1:],
                    device=input_video_tchw.device,
                    dtype=input_video_tchw.dtype,
                )
                interleaved[nvs_mask] = input_video_tchw
                interleaved[~nvs_mask] = nvs_target_tchw.to(device=input_video_tchw.device, dtype=input_video_tchw.dtype)

                with torch.no_grad():
                    _, _, camera_encs = spatial_encoder([interleaved.to(device)], return_cam_enc=True)
                    all_extrinsics_w2c, all_intrinsics = pose_encoding_to_extri_intri(
                        camera_encs[0][-1].unsqueeze(0), interleaved.shape[-2:]
                    )

                mask_dev = nvs_is_input_mask.to(device=all_extrinsics_w2c.device, dtype=torch.bool)
                input_extrinsics_w2c = all_extrinsics_w2c[:, mask_dev]
                input_intrinsics = all_intrinsics[:, mask_dev]
                target_extrinsics_w2c = all_extrinsics_w2c[:, ~mask_dev]
                target_intrinsics = all_intrinsics[:, ~mask_dev]

                out_pt.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema_version": "v1",
                    "pose_source": "vggt",
                    "video_key": video_key,
                    "video_path": video_abs_path,
                    "status": "ok",
                    "input_frame_indices": input_frame_idx_tensor.cpu(),
                    "novel_pool_indices": novel_pool_indices_tensor.cpu(),
                    "nvs_is_input_mask": nvs_is_input_mask.cpu(),
                    "input_extrinsics_w2c": input_extrinsics_w2c.cpu(),
                    "input_intrinsics": input_intrinsics.cpu(),
                    "target_extrinsics_w2c": target_extrinsics_w2c.cpu(),
                    "target_intrinsics": target_intrinsics.cpu(),
                    "intrinsics_ref_hw": [int(interleaved.shape[-2]), int(interleaved.shape[-1])],
                    # jj: Minimal reproducibility context for validation.
                    "sampling_config_min": {
                        "video_min_frames": args.video_min_frames,
                        "video_max_frames": args.video_max_frames,
                        "video_frame_fps": args.video_frame_fps,
                        "sampling_enforce_real_neighbour": bool(args.sampling_enforce_real_neighbour),
                        "neighbour_mode": args.neighbour_mode,
                        "neighbour_max_step": args.neighbour_max_step,
                    },
                }
                torch.save(payload, out_pt)

                item["status"] = "ok"
                stats["ok"] += 1

            except Exception as e:
                item["status"] = "failed"
                item["error"] = repr(e)
                stats["failed"] += 1

            item["elapsed_sec"] = round(time.time() - t0, 4)
            rep.write(json.dumps(item, ensure_ascii=False) + "\n")

            if (idx + 1) % 20 == 0:
                print(
                    f"[PROGRESS] {idx+1}/{len(records)} | ok={stats['ok']} "
                    f"skip={stats['skip_existing']} failed={stats['failed']}"
                )

    stats["elapsed_sec"] = round(time.time() - stats["start_time"], 2)
    stats.pop("start_time", None)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"[DONE] summary: {summary_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[DONE] report : {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
