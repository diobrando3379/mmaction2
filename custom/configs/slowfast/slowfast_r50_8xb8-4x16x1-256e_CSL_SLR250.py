_base_ = [
    '/home/diobrando/workspace/mmaction2/configs/_base_/models/slowfast_r50.py',
    '/home/diobrando/workspace/mmaction2/configs/_base_/default_runtime.py'
]
import json

load_from = 'custom/pre_pth/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'

ann_file_train = 'custom/data/CSL_SLR250_CUT/train_split.txt'  # 相对路径
ann_file_val   = 'custom/data/CSL_SLR250_CUT/val_split.txt'
ann_file_test  = 'custom/data/CSL_SLR250_CUT/test_split.txt'

# 若txt里用绝对路径，在此留空；若相对路径，在此写根目录
data_root = '/mnt/e/Dataset/CV_Dataset/CSL_500'
data_root_val = '/mnt/e/Dataset/CV_Dataset/CSL_500'
data_root_test = '/mnt/e/Dataset/CV_Dataset/CSL_500'
NUM_CLASSES = 250

model = dict(
    backbone=dict(
        resample_rate=8, # τ = T(fast) / T(slow)
        speed_ratio=8, # α = τ
        channel_ratio=8 # β -> 这是超参数，Fast 路每个层/阶段的通道数压缩为 Slow 路的 1/β
    ),
    cls_head=dict(num_classes=NUM_CLASSES)
)

file_client_args = dict(io_backend='disk')

# ===== RawFrameDataset 版本（SlowFast，先采样再帧解码）=====
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop', area_range=(0.5, 1.0), aspect_ratio_range=(0.9, 1.1)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=10, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

dataset_type = 'RawframeDataset'

train_dataloader = dict(
    batch_size=14,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),       # RawFrame 用 img 键
        filename_tmpl='{:06}.jpg',             # 按你的实际命名修改
        start_index=1,                         # 若从 000001.jpg 开始
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=14,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.jpg',
        start_index=1,
        pipeline=val_pipeline,
        test_mode=True)
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
        filename_tmpl='{:06}.jpg',
        start_index=1,
        pipeline=test_pipeline,
        test_mode=True)
)

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=256, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.0001, by_epoch=True, begin=0, end=9, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=256, eta_min=0, by_epoch=True, begin=10, end=256)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=1,                # 每 1 个迭代记录一次
        log_metric_by_epoch=True   # 验证集按 epoch 作为步轴
    ),
    checkpoint=dict(type='CheckpointHook', interval=4, max_keep_ckpts=2, save_best='auto')
)

log_processor = dict(
    type='LogProcessor',
    window_size=1,
    by_epoch=True,
    log_with_hierarchy=True
)