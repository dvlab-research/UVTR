_base_ = [
    '../camera/uvtr_c_r50_h11.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
cam_sweep_num = 4
fp16_enabled = True
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(cam_sweep_num=cam_sweep_num)

model = dict(
    img_backbone=dict(depth=101),
    pts_bbox_head=dict(
        view_cfg=dict(
            num_sweeps=cam_sweep_num,
            sweep_fusion=dict(type='sweep_cat_with_time'),
        ),
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='UniCrossAtten',
                            num_points=1,
                            embed_dims=256,
                            num_sweeps=cam_sweep_num,
                            fp16_enabled=fp16_enabled)]
        )))
    ))

train_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFiles', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='UnifiedRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFiles', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['img'])
]

data = dict(
    train=dict(pipeline=train_pipeline, modality=input_modality,),
    val=dict(pipeline=test_pipeline, modality=input_modality),
    test=dict(pipeline=test_pipeline, modality=input_modality))

load_from='ckpts/uvtr/pretrain/fcos3d.pth' # please download the pretrained model from the our git