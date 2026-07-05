_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/ir_centerpoint_voxel01_second_secfpn_kitti.py',
    '../_base_/schedules/cyclic-80e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -5, 70.4, 40, 3]
class_names = ['Pedestrian', 'Cyclist', 'Car']

data_prefix = dict(pts='training/velodyne_reduced')

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(
        bbox_coder=dict(pc_range=point_cloud_range[:2]),
        ir_head=dict(type='KeypointHeadAttention'),
        warmup_epochs=0),  # disable warmup for overfitting test
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'MinifiedKittiDataset'
data_root = 'data/kitti/'
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        num_samples=8,  # overfit test
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        num_samples=8,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_train.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=100, val_interval=5)
