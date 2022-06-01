_base_ = [
    '../camera/uvtr_c_r50_h11.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
pts_voxel_size = [0.1, 0.1, 0.2]
lidar_sweep_num = 10
cam_sweep_num = 3
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

distill_type = '3D_2D'
input_modality = dict(use_lidar=True, cam_sweep_num=cam_sweep_num)
fp16_enabled = True

model = dict(
    type='UVTRKDL',
    use_grid_mask=True,
    distill_type=distill_type,
    pretrained_img='ckpts/uvtr/pretrain/fcos3d.pth', # please download the pretrained model from the our git
    pretrained_pts='ckpts/uvtr/pretrain/uvtr_l_v01_h5.pth', # please download the pretrained model from the our git
    load_img=['img_backbone', 'img_neck'],
    load_pts=['pts_middle_encoder','pts_backbone','pts_neck'],
    img_backbone=dict(depth=101),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=pts_voxel_size,
        max_voxels=(90000, 120000),
        deterministic=False,
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoderHD',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        fp16_enabled=False), # not enable FP16 here
    pts_backbone=dict(
        type='SECOND3D',
        in_channels=[256, 256, 256],
        out_channels=[128, 256, 512],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 4],
        is_cascade=False,
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv3d', kernel=(1,3,3), bias=False)),
    pts_neck=dict(
        type='SECOND3DFPN',
        in_channels=[128, 256, 512],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv3d', bias=False),
        extra_conv=dict(type='Conv3d', num_conv=3, bias=False),
        use_conv_for_no_stride=True,
        use_for_distill=True),
    pts_bbox_head=dict(
        type='UVTRKDHead',
        kd_cfg=dict(
            type=distill_type,
            teacher_trans='raw',
            student_trans='none',
            position='before_relu',
            loss_dist='MSE_partial',
            loss_reduction='mean',
            loss_weight=1e-2,
        ),
        view_cfg=dict(
            num_sweeps=cam_sweep_num,
            sweep_fusion=dict(type='sweep_cat_with_time'),
            use_for_distill=True,
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
        ))))
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=lidar_sweep_num-1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadMultiViewMultiSweepImageFromFiles', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='UnifiedRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points', 'img'])
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

load_from=None