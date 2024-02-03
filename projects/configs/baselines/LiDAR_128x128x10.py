_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"
img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]   #10类
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False

dataset_type = 'NuscOCCDataset'  #/ projects/occ_plugin/datasets/nuscenes_occ_dataset.py
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='OccNet',             #/projects/occ_plugin/occupancy/detectors/occnet.py

    loss_norm=True,

    pts_voxel_layer=dict(
        max_num_points=10,    #每个体素中的最大点数
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint；[dx, dy, dz]
        max_voxels=(90000, 120000)), #每个体素化层xy方向上的最大体素

    pts_voxel_encoder=dict(   #返回每个体素所有点的特征均值(N,num_features)
        type='HardSimpleVFE', #编码器类型/root/mmdetection3d/mmdet3d/models/voxel_encoders/voxel_encoder.py
        num_features=5),      #特征数量

    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',  #/OpenOccupancy/projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py
        input_channel=4,    
        base_channel=16,
        out_channel=numC_Trans,    #80
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024, 1024, 80],  # hardcode, xy size follow centerpoint
        ),

    occ_encoder_backbone=dict(
        type='CustomResNet3D',  #/projects/occ_plugin/occupancy/backbones/resnet3d.py
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',          #/projects/occ_plugin/occupancy/necks/fpn3d.py
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',       #/projects/occ_plugin/occupancy/dense_heads/occ_head.py
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=1), #10
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['gt_occ', 'points'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]


test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

train_config=dict(
        type=dataset_type,           #NuscOCCDataset
        data_root=data_root,         #data/nuscenes/
        occ_root=occ_path,           #./data/nuScenes-Occupancy
        ann_file=train_ann_file,     #./data/nuscenes/nuscenes_occ_infos_train.pkl
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,     #use_lidar
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,           #[512, 512, 40]
        pc_range=point_cloud_range,  #[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        box_type_3d='LiDAR'),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

