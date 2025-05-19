auto_scale_lr = dict(base_batch_size=16)
backend_args = None
data_root = 'C:\\Users\\LEGION\\Desktop\\Data\\'
dataset_type = 'DualBranchCocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
    logger=dict(interval=50, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook'))
default_scope = None
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, type='mmengine.runner.LogProcessor', window_size=50)
max_epochs = 12
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='torchvision://resnet50',
            type='mmengine.model.weight_init.PretrainedInit'),
        norm_cfg=dict(
            requires_grad=False,
            type='torch.nn.modules.batchnorm.BatchNorm2d'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.models.ResNet'),
    bbox_head=dict(
        loss_bbox=dict(
            loss_weight=5.0, type='mmdet.models.losses.smooth_l1_loss.L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.models.losses.focal_loss.FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(
            loss_weight=2.0, type='mmdet.models.losses.iou_loss.GIoULoss'),
        num_classes=80,
        sync_cls_avg_factor=True,
        type='mmdet.models.DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.models.DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(
            num_groups=32, type='torch.nn.modules.normalization.GroupNorm'),
        num_outs=4,
        out_channels=256,
        type='mmdet.models.ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='mmdet.models.task_modules.FocalLossCost',
                    weight=2.0),
                dict(
                    box_format='xywh',
                    type='mmdet.models.task_modules.BBoxL1Cost',
                    weight=5.0),
                dict(
                    iou_mode='giou',
                    type='mmdet.models.task_modules.IoUCost',
                    weight=2.0),
            ],
            type='mmdet.models.task_modules.HungarianAssigner')),
    type='mmdet.models.DualBranchDINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        lr=0.0001, type='torch.optim.adamw.AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            11,
        ],
        type='mmengine.optim.scheduler.lr_scheduler.MultiStepLR'),
]
resume = False
test_cfg = dict(type='mmengine.runner.loops.TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations\\annotations_val.json',
        backend_args=None,
        data_root='C:\\Users\\LEGION\\Desktop\\Data\\',
        img_prefix_amp='val\\IT\\amplitude\\',
        img_prefix_phase='val\\IT\\phase\\',
        pipeline=[
            dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='mmdet.datasets.transforms.Resize'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        test_mode=True,
        type='DualBranchCocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
test_evaluator = dict(
    ann_file=
    'C:\\Users\\LEGION\\Desktop\\Data\\annotations\\annotations_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='mmdet.evaluation.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type='mmdet.datasets.transforms.Resize'),
    dict(type='mmdet.datasets.transforms.LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.datasets.transforms.PackDetInputs'),
]
train_cfg = dict(
    max_epochs=12,
    type='mmengine.runner.loops.EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='mmdet.datasets.AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations\\annotations_train.json',
        backend_args=None,
        data_root='C:\\Users\\LEGION\\Desktop\\Data\\',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        img_prefix_amp='train\\IT\\amplitude\\',
        img_prefix_phase='train\\IT\\phase\\',
        pipeline=[
            dict(
                backend_args=None,
                type='mmcv.transforms.loading.LoadImageFromFile'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(prob=0.5, type='mmdet.datasets.transforms.RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            resize_type='mmdet.datasets.transforms.Resize',
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='mmcv.transforms.RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            resize_type='mmdet.datasets.transforms.Resize',
                            scales=[
                                (
                                    400,
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
                                ),
                            ],
                            type='mmcv.transforms.RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='mmdet.datasets.transforms.RandomCrop'),
                        dict(
                            keep_ratio=True,
                            resize_type='mmdet.datasets.transforms.Resize',
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='mmcv.transforms.RandomChoiceResize'),
                    ],
                ],
                type='mmcv.transforms.RandomChoice'),
            dict(type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        type='DualBranchCocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.sampler.DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmcv.transforms.loading.LoadImageFromFile'),
    dict(type='mmdet.datasets.transforms.LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='mmdet.datasets.transforms.RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    resize_type='mmdet.datasets.transforms.Resize',
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='mmcv.transforms.RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    resize_type='mmdet.datasets.transforms.Resize',
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='mmcv.transforms.RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='mmdet.datasets.transforms.RandomCrop'),
                dict(
                    keep_ratio=True,
                    resize_type='mmdet.datasets.transforms.Resize',
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='mmcv.transforms.RandomChoiceResize'),
            ],
        ],
        type='mmcv.transforms.RandomChoice'),
    dict(type='mmdet.datasets.transforms.PackDetInputs'),
]
val_cfg = dict(type='mmengine.runner.loops.ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations\\annotations_val.json',
        backend_args=None,
        data_root='C:\\Users\\LEGION\\Desktop\\Data\\',
        img_prefix_amp='val\\IT\\amplitude\\',
        img_prefix_phase='val\\IT\\phase\\',
        pipeline=[
            dict(backend_args=None, type='mmcv.transforms.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='mmdet.datasets.transforms.Resize'),
            dict(
                type='mmdet.datasets.transforms.LoadAnnotations',
                with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.datasets.transforms.PackDetInputs'),
        ],
        test_mode=True,
        type='DualBranchCocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'C:\\Users\\LEGION\\Desktop\\Data\\annotations\\annotations_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='mmdet.evaluation.CocoMetric')
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.visualization.DetLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
work_dir = './work_dirs\\my_dino_4scale_r50_8xb2_12e_coco'
