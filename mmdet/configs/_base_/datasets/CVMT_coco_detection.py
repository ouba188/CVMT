from mmcv.transforms import RandomChoice, RandomChoiceResize
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomCrop, RandomFlip, Resize)
from mmengine.dataset.sampler import DefaultSampler
from mmdet.datasets import AspectRatioBatchSampler, CVMTCocoDataset
from mmdet.datasets.transforms import LoadDataFromFile_CVMT
from mmdet.evaluation import CocoMetric

# Dataset settings
# Use the custom COCO-style dataset class for CVMT
dataset_type = CVMTCocoDataset
# Root directory for dataset files (annotations, images)
# data_root = '/home/wangpengcheng/Desktop/Data/'  # opensarship2.0
data_root = '/media/sata/wpc/Dataf/'  # FAIR-CSAR
backend_args = None  # No special backend config (e.g., for remote storage)

# Training data pipeline:
train_pipeline = [
    # Load both amplitude and phase image pairs from disk
    dict(type=LoadDataFromFile_CVMT, scale_factor=(1.0, 1.0), backend_args=backend_args),
    # Load bounding box annotations
    dict(type=LoadAnnotations, with_bbox=True),
    # Random horizontal flip with probability 0.5
    dict(type=RandomFlip, prob=0.5),
    # Apply one of two resize/crop strategies at random
    dict(
        type=RandomChoice,
        transforms=[
            [
                # Strategy A: randomly resize to one of the preset scales
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize,
                    scales=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                            (736, 1024), (768, 1024), (800, 1024)],
                    keep_ratio=True
                )
            ],
            [
                # Strategy B: resize to a large long side, crop, then resize again
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize,
                    scales=[(400, 3300), (500, 3300), (600, 3300)],
                    keep_ratio=True
                ),
                dict(
                    type=RandomCrop,
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=False,
                ),
                dict(
                    type=RandomChoiceResize,
                    resize_type=Resize,
                    scales=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                            (736, 1024), (768, 1024), (800, 1024)],
                    keep_ratio=True
                )
            ]
        ]
    ),
    # Pack the processed images and annotations into the format expected by the model
    dict(type=PackDetInputs)
]

# Testing/validation data pipeline (no augmentation)
test_pipeline = [
    dict(type=LoadDataFromFile_CVMT, backend_args=backend_args),  # Load images
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),   # Resize to fixed size
    dict(type=LoadAnnotations, with_bbox=True),               # Load ground truth boxes (for eval)
    dict(type=PackDetInputs)                                  # Pack inputs
]

# Training data loader configuration
train_dataloader = dict(
    batch_size=2,               # Two image pairs per batch
    num_workers=8,              # Data loading workers
    persistent_workers=True,    # Keep workers alive between epochs
    sampler=dict(type=DefaultSampler, shuffle=True),  # Shuffle dataset each epoch
    batch_sampler=dict(type=AspectRatioBatchSampler),  # Group images by aspect ratio
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

# Validation data loader configuration
val_dataloader = dict(
    batch_size=1,              # Single image pair per batch for evaluation
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),  # No shuffling during validation
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/annotations_val.json',
        data_prefix=dict(
            img_prefix_amp='val/IT',     # Path prefix for amplitude images
            img_prefix_phase='val/POS'    # Path prefix for phase images
        ),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        test_mode=True,             # Enable test mode to skip some train-only steps
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# Use the same loader for test as for validation
test_dataloader = val_dataloader

# COCO-style evaluator for validation and testing
val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/annotations_val.json',
    metric='bbox',             # Evaluate bounding box metrics (mAP)
    format_only=False,
    backend_args=backend_args
)

test_evaluator = val_evaluator  # Reuse validation evaluator settings for testing