_base_ = [
    './uvtr_lidar_base.py'
]

point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
pts_voxel_size = [0.075, 0.075, 0.2]
voxel_size = [0.15, 0.15, 8]
lidar_sweep_num = 10


# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

model = dict(
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=pts_voxel_size,
    ),
    pts_middle_encoder=dict(
        sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
        bbox_coder=dict(
            pc_range=point_cloud_range,
            voxel_size=voxel_size), 
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[720, 720, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        assigner=dict(
            pc_range=point_cloud_range))),
    
)

data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')
db_sampler = dict(
    type='UnifiedDataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'nuscenes_unified_dbinfos_train.pkl', # please change to your own database file
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

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
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='UnifiedObjectSample', db_sampler=db_sampler), # commit this for the last 2 epoch
    dict(
        type='UnifiedRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='UnifiedRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points'])
]
test_pipeline = [
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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['points'])
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     # Add double-flip augmentation
    #     flip=True,
    #     pcd_horizontal_flip=True,
    #     pcd_vertical_flip=True,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D', sync_2d=False),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points'])
    #     ])
]


optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))