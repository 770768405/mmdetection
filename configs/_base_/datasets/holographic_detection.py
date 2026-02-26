# dataset settings for holographic imaging
dataset_type = 'HolographicDataset'
data_root = 'data/holographic/'

# Normalization config - can be adjusted based on holographic image characteristics
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Training pipeline with holographic image loading
train_pipeline = [
    dict(
        type='LoadHolographicImage',
        to_float32=False,
        color_type='color',
        load_depth=True,  # Enable depth map loading
        load_phase=False  # Disable phase map loading (enable if needed)
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# Test pipeline
test_pipeline = [
    dict(
        type='LoadHolographicImage',
        to_float32=False,
        color_type='color',
        load_depth=True,
        load_phase=False
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Dataset configuration
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/',
        depth_prefix=data_root + 'train_depth/',  # Optional depth maps
        phase_prefix=data_root + 'train_phase/',  # Optional phase maps
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        depth_prefix=data_root + 'val_depth/',
        phase_prefix=data_root + 'val_phase/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'test/',
        depth_prefix=data_root + 'test_depth/',
        phase_prefix=data_root + 'test_phase/',
        pipeline=test_pipeline))

# Evaluation settings
evaluation = dict(interval=1, metric='bbox')
