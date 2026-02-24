


story:
sample frame mattters (3D q in general n actual spatil), while PE esepeically need to be take care.

fot THW style, we extend with P.
fot HW style, we extend with P.

-pose from other nets. or even idealy online? or from slam?
-samples from other sampling stategies
-base PE format: THW and HW

24.02
TODO:
train a spmllm wo connector model for fair comparision?
we compete with standard spmllm traning to show that we only need pose?

23.02 
train: qwen3 PHW_1st vs qwen3
3002675 3002421 3001694 _qwen3_phw_1st_2x8_hpc_50ViCA
3002955 _qwen3_2x8_hpc_50ViCA
eval: qwen3+PHW_1st on effi
3001913
3001918
3001919

8 16 32
eval sft model
3002689 -PTHW1st-skipCnc-50k-sft
3002697 -spmllm-baseline-50k-sft

2995208 train100 _PTHW_medoid_882424BUGFIXED_ACTUAL_skipCnc_2x8_hpc_100ViCA
TODO: qwen3 PHW
TODO: online trace selcted frame n reapated
TODO: time merge set to 1? hwo to fit in story?


EVAL: PHW only+effi+sp mllm(with connector)
EVAL: QWEN3 enforce repeat
2991093
EVAL:PTHW+effi+sp mllm(with connector)
2991090 2991091 2991092 1st
2991088 modroid
EVAL: PHW?

TRAIN:PTHW

2990609 _PTHW_1st_882424BUGFIXED_ACTUAL_skipCnc_2x8_hpc
2991085 _PTHW_medoid_882424BUGFIXED_ACTUAL_skipCnc_2x8_hpc
TRAIN OTHER FREQ ALLOCATION?


22.02
offline reorder the index in : efficient and fps--n reeval all
duplacate time sa as justify
offline of PTHW(1st/modroid)--16

21.02
TODO:
check from adapted to see if large scaled is better?



eval: SPMLLM PTHW on fps_stdnorm_1st_sampling
2985141 f8
2985142 f16
2985143 f32
eval: SPMLLM PTHW on fps_stdnorm_medoid_sampling
2985156 f16


eval:fps_stdnorm_medoid_sampling
2983833 qwen3 f8
2983834 qwen3 f16
2983835 qwen3 f32
eval:efficient_sampling_v0_hybrid
2983836 qwen3 f8
2983837 qwen3 f16
2983838 qwen3 f32
eval:fps_stdnorm_medoid_sampling
2983840 spatialmllm f8/16/32
eval:efficient_sampling_v0_hybrid
2983841 spatialmllm f8/16/32



offline reorder the index in : efficient and fps
RUN:
train: pica basic-50
2978340 nan X2978135 _baseline_spmllm_2x8_hpc: -spmllm
ours: no connector. just PTHW.
2990609 _PTHW_1st_882424BUGFIXED_ACTUAL_skipCnc_2x8_hpc

#PTHW all wrong having bug of freq allocation
2984046 _PTHW_1st_ACTUAL_skipCnc_2x8_hpc -ours spmllm: PTHW 1st  --mm_connector false ; skip_connector True
2984048 _PTHW_medoid_ACTUAL_skipCnc_2x8_hpc -ours spmllm: PTHW 1st   --mm_connector false ; skip_connector True
2978337 _PTHW_medoid_skipCnc_2x8_hpc -ours spmllm: PTHW mdroid --infact still USE CONNECTOR
(stoped)2978339 2978132 _PTHW_1st_skipCnc_2x8_hpc -ours spmllm: PTHW 1st  ---infact still USE CONNECTOR
X 2978136 _baseline_qwen25_skipCnc_2x8_hpc: -qwen2.5 

-ours spmllm: PTHW 1st pica-full 


20.02
Eval:
2949219 qwen3 fps32
2950453 sp mllm time_aware_Sa f8
2950461 sp mllm time_aware_Sa + adapted f8
2950467 sp mllm time_aware_Sa + adapted f16
2950468 sp mllm time_aware_Sa + adapted f32

Gen:
2949458 effi_hybrid 64
2950017 fps_mediod_stdnorm_8
2950018 fps_mediod_stdnorm_16
2950019 fps_mediod_stdnorm_32


18.02
EVAL:
adapted on mllm
effi: 8 16
sa: 8 16
wo T on mllm

our trained on sqa40k:
8/16/32
custom mllm 2914804
mllm 2914805

effi: 
qwen3 32
mllm 32

sp mllm 1.1 135:
2912404 fps 16 
2913112 fps 8
2913111 eff 16
2912414 eff 8
2912408 sa 8

qwen3vl-rp-routeplan-missing
2912431 f8
2912432 f16 
2912436 f32

qwen2.5: 
2910786 fps 16
2911430 ?2910813 fps 8
2911432 ?2910838 effi 16 
2910851 effi 8
qwen3: 
2911079 fps 16 41069025
2911434 ?2911080 effi 16

GEN:
# eff16 :2912355
# effi32:2912384
# fps8:2912390
# fps32:2912393
//fps:
f16:2909474
f8:2909475
//effecient:
?f16:2909833
f8:2909834


16.02:
4GPU prope-spatailmllm + sqa3d
4GPU spatailmllm + sqa3d

2867904: 2GPU prope-spatailmllm + sqa3d
2892391: By chance successed (no idea why) 2868429 x2867906: 2GPU spatailmllm + sqa3d

prope-spatailmllm  + vsibench_test
spatailmllm  + vsibench_test

15.02:
2853405 tma_sa16_adapted_spbaseline
 tma_sa16_prope_spbaseline


2850844 eval sa16_prope_spbaseline
2850847 eval sa32_prope_spbaseline
2850856 eval tma_sa16_spbaseline
2850857 eval tma_sa32_spbaseline
2850901 eval tma_sa16_prope_spbaseline
2850902 eval tma_sa32_prope_spbaseline


2844454 mergeaware_sa_f16
2849496 eval mergeaware_sa_f16

14.02:



todo:
eval 16 32
2841955 mergeaware_uniform 16
2841956 mergeaware_uniform 32

gen:
2839266 16: mergeaware_uniform  scannetpp
2839356 32: mergeaware_uniform  scannet

eval:
2838125 mergeaware_sa_f8
2838126 mergeaware_sa_f16
2838128 mergeaware_sa_f32
2838117 sa8

eval: merge_uni_f8
2835055

13.02:
gen the mergeaware_sa/mergeaware_uniform
mergeaware_sa: the ratio of the overlapp are linear increase with nframes increase. check the .err can do stats on it. 
D 8 2834355 2828004
D 16 2834358  2828006
D 32 2834360 2828008
mergeaware_uniform: 
D: 8 2828009
X 16 2834364 2828010
X 32 2834365 2828011
mergeaware_uniform failed since one sample failed:
16: (infact it is the 1st sample in scannetpp)
09c1414f1b
scannetpp
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/sw_sampling_09c1414f1b_gpu0_jf8y3d91/frame_1287.png'
[GPU 0] Error processing 09c1414f1b: [Errno 2] No such file or directory: '/tmp/sw_sampling_09c1414f1b_gpu0_jf8y3d91/frame_1287.png'

32:(failed since scene0697_01)
scannet
scene0697_01
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/sw_sampling_scene0697_01_gpu0_pw7ojscb/frame_1751.png'
/data/horse/ws/jixu233b-metadata_ws/datasets/vsibench/mergeaware_uniform_sampling_32f_rnd_fidss30/scannet/scene0697_01

12.02
#eval qwen3vl with video input for frames num on sa
sa8 2834638 2828052 2808207
sa16 2808218 2804290
sa32 2804296

#eval qwen3vl with video input for frames num
f8 2808110
f16 2804235
f32 2804225

#Gen sa samples cam pose and other meta
2835137 2828283 2823358(sa only) 2803812 f8_both extra
2803826 f16_both extra
2806350 f32_both extra


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