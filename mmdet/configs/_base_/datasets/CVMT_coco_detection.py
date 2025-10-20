from mmcv.transforms import RandomChoice, RandomChoiceResize
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs_Dual,
                                       RandomCrop_Dual, RandomFlip, Resize_Dual)
from mmengine.dataset.sampler import DefaultSampler
from mmdet.datasets import AspectRatioBatchSampler, CVMTCocoDataset
from mmdet.datasets.transforms import LoadDataFromFile_CVMT
from mmdet.evaluation import CocoMetric

# Use the custom COCO-style dataset class for CVMT (OpenSARShip and Fair-CSAR)
dataset_type = CVMTCocoDataset
data_root = '/path/to/the/data/root/'
backend_args = None


train_pipeline = [
    dict(type=LoadDataFromFile_CVMT, scale_factor=(1.0, 1.0), backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomChoice,
        transforms=[
            [
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize_Dual,
                    scales=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                            (736, 1024), (768, 1024), (800, 1024)],
                    keep_ratio=True
                )
            ],
            [
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize_Dual,
                    scales=[(400, 3300), (500, 3300), (600, 3300)],
                    keep_ratio=True
                ),
                dict(
                    type=RandomCrop_Dual,
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=False,
                ),
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize_Dual,
                    scales=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                            (736, 1024), (768, 1024), (800, 1024)],
                    keep_ratio=True
                )
            ]
        ]
    ),
    dict(type=PackDetInputs_Dual)
]


test_pipeline = [
    dict(type=LoadDataFromFile_CVMT, backend_args=backend_args),
    dict(type=Resize_Dual, scale=(800, 800), keep_ratio=True),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=PackDetInputs_Dual)
]


train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/annotations_train.json',
        data_prefix=dict(
            img_prefix_amp='train/IT',     # Path prefix for amplitude images
            img_prefix_phase='train/POS'    # Path prefix for phase images
        ),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/annotations_val.json',
        data_prefix=dict(
            img_prefix_amp='val/IT',     # Path prefix for amplitude images
            img_prefix_phase='val/POS'    # Path prefix for phase images
        ),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/annotations_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

test_evaluator = val_evaluator