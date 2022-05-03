# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='Siamese_residual',
        depth=34,
        num_stages=4,
        out_indices=(0,1 ,2,3),
        frozen_stages=1 ,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,              # 类别数
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


# dataset settings
dataset_type = 'CocoDataset'
data_root =  './data/pcb_coco/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadtwoImages', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='TwoPhotoMetricDistortion'),
    dict(type='TwoResize', img_scale=(1000,1000), keep_ratio=False),
    dict(type='TwoRandomFlip', flip_ratio=0.5),
    dict(type='TwoNormalize', **img_norm_cfg),
    dict(type='TwoPad', size_divisor=32),
    dict(type='TwoDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadtwoImages'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000,1000),
        flip=False,
        transforms=[
            dict(type='TwoResize', keep_ratio=True),
            dict(type='TwoRandomFlip'),
            dict(type='TwoNormalize', **img_norm_cfg),
            dict(type='TwoPad', size_divisor=32),
            dict(type='TwoDefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/train.json',
            img_prefix=data_root + 'train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.0001,
    step=[24, 28])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric=['bbox'])
find_unused_parameters = True

checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]