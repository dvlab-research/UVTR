_base_ = [
    '../uvtr_camera_base.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
bev_stride = 4
sample_num = 11
voxel_shape = [int(((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])//bev_stride),
               int(((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])//bev_stride),
               sample_num]


model = dict(
    pts_bbox_head=dict(
        view_cfg=dict(
            num_points=sample_num,
            voxel_shape=voxel_shape,
        ),
    ))