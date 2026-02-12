Eval Exp: task. room size estimation on all arkit. 150 questions.
0. spatialmllm sft: default.
'all': {'micro': 0.5320000018676122, 'macro': 0.5320000018676122}
1. spatialmllm sft: 2Dfeature_only (wo fusion, mRoPE). 
2. spatialmllm sft: 3Dfeature_only (wo fusion, mRoPE). 

3. Qvwen vl2.5: (2D only, mRoPE)
'all': {'micro': 0.29066666692495347, 'macro': 0.29066666692495347}


TODO:
2D token MLP to half; 3D token the other half. Use CAT rather fuse.
Then for the 2D half we conduct PRoPE.

second_per_grid_ts/tokens_per_second: set to 1 in sft spatialmllm (25 in qwen2.5vl)
temporal_patch_size: set to 2 both for qwem2.5 and spatialmllm
this explain why: sft_spatiallmllm need get_rope_index_2 rather get_rope_index_25 (default). sft_spatiallmllm is optimized in a way, that candidate frames are having large view differs! (considering temporal_patch_size still 2, 'the merging' is wrong! )

- the way 2D vl fuse is only somehow safe when coupled with uniform sampling: alwasy consenctive 2 samples. As it can be risky if you sample optimal multiview img from video rather (the merge two imgs have large view changes).
- VGG infact give u view independent world repre per frame, when merge frame, it can be more like enlarging the mapping.
-Consider the above, the fusing connector in spatailmlllm is suspicious.

- Here(https://github.com/liruilong940607/prope/issues/11) regaring PRoPE has more intersting regaridng Time dim. 
    Our sweet pots:
    - we can alternating xy(or same modality spatial: xyz for pointcloud)
    - we can think about whether to apply Pose for HighFreq or Low Freq (what about geodist alibi calib?)
    - is there room for VO-RoPE. it is along the finding of GTA.


- CAPE: SE3 only PE is theoretically ONLY condition on camPose.
- notice the author is obtaining the mvg embedding per frame from VGGT already (implecit contain cam info). Maybe no need to condition on cam again? Along the line, set time dim all the same maybe have positive effect?
- dynamic: differ the time to approximate the token similarity when self attn
- static scene at different timepoint: how much to correlative should only relates the cam pose inrellecant to the frame order? set the same?

- argue: the video MLLM in fact should condition on viewperspectives rather frame_id.
    - evidence: performance on spatail reasoning tasks not changes.
    - if we ondition on viewperpseticves (use trans, rot, or alone)
    - what about alibi but in angle_axis_dis(p1-p2) style!
- viewmats: torch.Tensor,  # (batch, cameras, 4, 4):
    various scaling strategy.
    how to choose reference frame, does this matter?: DA3.
- DOES scales in Views are consistent?
- Viewmats: is trans is in 100 magnitude, it is numerical not stable
- Camera K: accuracy in K
- w2c presentation.
- RoPE require the cx cy follow img size.
- extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:]) #w2c format, JJ confirmed in VGGT source code.
- Maybe lastly we can completely abadon VGGT spatial: just use cam token?
    # fuse video and spatial embeddings
    fused_embeds = self.connector(
        image_embeds=image_embeds,
        spatial_embeds_list=spatial_embeds_list, #repeat with image_embeds 
        patch_start_idx=patch_start_idx,
        grid_thw=image_grid_thw,
    )