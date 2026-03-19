# path：/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/SPMLLM-DATA/precompute_pose_vggt_16_pa

# setting：
- min max f16
- enforce nbr sampling: after; step1;
- the base_interval was set to 4 (now we adjust the default as 2 for furtual gen. if needed) in this version, (is set to 2 in online mode). This value has littl affect and we regard online and precompute samples being the same:
            
            interval = getattr(self.data_args, "base_interval", 4)
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )