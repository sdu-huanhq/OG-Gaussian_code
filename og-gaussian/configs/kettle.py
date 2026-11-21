
from configs._base_.default_runtime import hooks

_base_ = ["_base_/default_runtime.py"]

# misc custom setting
epoch = 2000
batch_size = 48  # bs: total bs in all gpus
num_worker = 24
dataset_repeat = 2
mix_prob = 0
empty_cache = False
enable_amp = False
find_unused_parameters = True
evaluate_freq = 100
val_freq = 50
max_grid_count = 2560
gs_activated=False
error_type='KettleError'
qf_normal=True

render_params = dict(
    num_views=4,
    render_size=128,
    dssim_weight=0.2,
    radius_scale=5.0,
    radius_percentile=95.0,
    K=((414.5749, 0.0, 256.0),
        (0.0, 414.5749, 256.0),
        (0.0, 0.0, 1.0)),
    merge_gap_margin=1.25,
    merge_gap_min_scale=4.0,
    merge_gap_min_abs=1.0,
)

hooks[5] = dict(type="CheckpointSaver", save_freq=evaluate_freq)

# dataset settings
dataset_type = "RepeatedDataset"
data_root = "dataset/generated_tasks_v2_gs_source/Kettle/Kettle-lift_kettle_and_pour_water_v5/gs_dataset_h5_new"
clip_emb_path = "data/clip_emb/v3"

in_channels = 14

collect_cfg = dict(
    type="CollectGS", 
    anc_keys=['encoding'], 
    act_keys=['encoding'],
    gs_dim=in_channels
)

network = dict(
    type="SedOed_LitePT_QueryField", #! scene encoder-decoder + object encoder-decoder
    qf_normal=qf_normal,
    scene_encoder_params=dict(
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2),
        enc_depths=(2, 2, 2),
        enc_channels=(72, 144, 252),
        enc_num_head=(4, 8, 14),
        enc_patch_size=(1024, 1024, 1024),
        enc_conv=(True, True, False),
        enc_attn=(False, False, True),
        enc_rope_freq=(100.0, 100.0, 100.0),
        dec_depths=(0, 0),
        dec_channels=(144, 252),
        dec_num_head=(8, 14),
        dec_patch_size=(1024, 1024),
        dec_conv=(False, False),
        dec_attn=(False, False),
        dec_rope_freq=(100.0, 100.0),
    ),
    object_net_params=dict(
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2),
        enc_depths=(2, 2, 2),
        enc_channels=(72, 144, 252),
        enc_num_head=(4, 8, 14),
        enc_patch_size=(1024, 1024, 1024),
        enc_conv=(True, True, False),
        enc_attn=(False, False, True),
        enc_rope_freq=(100.0, 100.0, 100.0),
        dec_depths=(0, 0),
        dec_channels=(144, 252),
        dec_num_head=(8, 14),
        dec_patch_size=(1024, 1024),
        dec_conv=(False, False),
        dec_attn=(False, False),
        dec_rope_freq=(100.0, 100.0),
    ),
    backbone_params=dict(
        in_skill_channels=48,
        use_skill=False,
        out_channels=6,
        width=144,
        depths=6,
        num_heads=8,
        rel_hidden_dim=64,
        rel_depth=6,
        rel_alpha_init=0.5,
    ),
    query_field_params=dict(
        property=["density", "gradient"],
        qf_normalize=None
    ))


# model settings
model = dict(
    type="FMGSDualPT_SedOed_render",
    num_inference_steps=100,
    network=network,
    epoch=epoch,
    vT_weight=1.0,
    vR_weight=0.5,
    render_weight=[0.5,],
    error_calculator=dict(type=error_type),
    trans_frame="spatial",
    rot_frame="spatial",
    source_init="se3_plus_gs",
    source_se3_trans_std=0.02,
    source_se3_rot_std_deg=5.0,
    source_gs_pos_std=0.002,
    source_gs_rot_std_deg=0.0,
    **render_params,
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.001, weight_decay=1.0e-04)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.02,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]


data = dict(
    train=dict(
        activated=gs_activated,
        type=dataset_type,
        repeat=dataset_repeat,
        data_root=data_root,
        clip_emb_path=clip_emb_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # action
            dict(type="RandomRotate", angle=[-1, 1], axis="z", p=1, apply_target='action'),
            dict(type="RandomRotate", angle=[-1 / 18, 1 / 18], axis="x", p=1, apply_target='action'),
            dict(type="RandomRotate", angle=[-1 / 18, 1 / 18], axis="y", p=1, apply_target='action'),
            dict(type="RandomShift", shift=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)), apply_target='action'),
            # anchor
            dict(type="RandomRotate", angle=[-1, 1], axis="z", p=1, apply_target='anchor'),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=1, apply_target='anchor'),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=1, apply_target='anchor'),
            # object-scene
            dict(type="RandomScale", scale=[0.8, 1.2], is_activated=gs_activated),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.005),
            dict(type="RandomDropout", dropout_ratio=[0, 0.2], dropout_application_ratio=0.3),
            dict(
                type="GridSampleGS1",
                min_grid_size=0.001,
                max_grid_count=max_grid_count,
                grid_size_increment=0.001,
                hash_type="fnv",
                mode="train",
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="GSNormalizeShared",
                grid_resolution=1024,
                is_activated=gs_activated,
            ),
            collect_cfg,
            dict(type="ToTensor"),
        ],
    ),
    val=dict(
        activated=gs_activated,
        type=dataset_type,
        split="test",
        data_root=data_root,
        clip_emb_path=clip_emb_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSampleGS1",
                min_grid_size=0.001,
                max_grid_count=max_grid_count,
                grid_size_increment=0.001,
                hash_type="fnv",
                mode="test",
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="GSNormalizeShared",
                grid_resolution=1024,
                is_activated=gs_activated,
            ),
            collect_cfg,
            dict(type="ToTensor"),
        ],
    ),
    test=dict(
        activated=gs_activated,
        type=dataset_type,
        split="test",
        data_root=data_root,
        clip_emb_path=clip_emb_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSampleGS1",
                min_grid_size=0.001,
                max_grid_count=max_grid_count,
                grid_size_increment=0.001,
                hash_type="fnv",
                mode="test",
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="GSNormalizeShared",
                grid_resolution=1024,
                is_activated=gs_activated,
            ),
            collect_cfg,
            dict(type="ToTensor"),
        ],
    ),
    demo=dict(
        activated=gs_activated,
        data_root=data_root,
        clip_emb_path=clip_emb_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSampleGS1",
                min_grid_size=0.001,
                max_grid_count=max_grid_count,
                grid_size_increment=0.001,
                hash_type="fnv",
                mode="test",
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="GSNormalizeShared",
                grid_resolution=1024,
                is_activated=gs_activated,
            ),
            collect_cfg,
            dict(type="ToTensor"),
        ],
    ),
)
