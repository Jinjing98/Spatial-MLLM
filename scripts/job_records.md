TODO:
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