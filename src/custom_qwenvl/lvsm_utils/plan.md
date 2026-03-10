Update the a new varient of 'custom-mllm' model --- control with 2 flags:
- enforce_LVSM
- LVSM_type: decoder-only (place holder for encoder-decoder). use pretrained weights of the model.
- NVS_loss_weight: 1.0 (not compute if it is small or equal to 0.0)

Compared to current 'custom-mllm' design the core diff:

Network Core:
 - Qwen2 vit token fused with lvsm patch embedding.
 - the geometry_vit (vggt head) need to be retained to abtain the camera intrinsics/extrincis, from which we can construct the raymap needed for lvsm (decoder-only version) token embedding.
 - the pipeline should strictly follow this design: src/custom_qwenvl/lvsm_utils/image.png

Critical point to checkout:
 - convert intrisics and pose_extrinsics (c2w or w2c?) to raymap

critical hyperparam:
 - num of qwen_visual_input_views: the current 16 frames used to conduct vlm QA. input frames for self.visual
 - num of target_views for LVSM decoding: 
    set num_of_input_frames-1 as default; and for each, randomly sample a frame from the intersection between each pair of input neighbouring frames.
 - num of spatial_input_views for spatial_encoder: considering we also need cam intrisic n pose for target views when decoding, for efficeinty, i suggested VGGT take  target_views+qwen_visual_input_views and run forward once?

 (the allocation of test frames random target frames i suggested should be precomputed in the dataloader? so dataloder should loading me extra part of target_views images)

Loss design: NVS_loss_weight
- current loss is only Cross Entropy for Instruct tuning.
- i need u to decoder the target view and regress the target images. then conduct NVS.

TAKE AWAY:
- do not intruduce new learnble param(except self.connector_lvsm for token_merging). all you need is in LVSM. just use properly layers as i have drafted in the figure.
- write critical n necessary unit test for each function u have implemented; show me strong evidence that u are correct in implmentation.
-the only new thing u need to implment: token merging of qwenvl from self.visual and CORRESPONDING input view tokens from pretrained LVSM.
for this part please first understand how self.visual process input frames, and learn how current self.connector merge tokens from two sources.
- comment with token dim when necessary.

Copy or import (reuse) related utils functions in LVSM rather implement on ur own.

Correct me if the above plan are not the best.
Ask me when u need to make designing. I need 100 control and transparancy for all the decision.