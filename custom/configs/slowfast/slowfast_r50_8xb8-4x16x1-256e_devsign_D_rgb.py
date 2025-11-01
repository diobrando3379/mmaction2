_base_ = [
    '/home/diobrando/workspace/mmaction2/configs/_base_/models/slowfast_r50.py',
    '/home/diobrando/workspace/mmaction2/configs/_base_/default_runtime.py'
]
import json
load_from = '/home/diobrando/workspace/mmaction2/custom/pre_pth/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'
# 参考：VideoDataset 标注格式说明（每行“视频路径 标签”）
ann_file_train = 'custom/data/devsign_D/train_list.txt'
ann_file_val   = 'custom/data/devsign_D/val_list.txt'
ann_file_test  = 'custom/data/devsign_D/test_list.txt'

# 若txt里用绝对路径，在此留空；若相对路径，在此写根目录
data_root = ''
data_root_val = ''

NUM_CLASSES = 500

# ===== 模型：覆盖类别数与归一化到 data_preprocessor =====
model = dict(
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'
    ),
    cls_head=dict(num_classes=NUM_CLASSES)
)

# ===== 文件后端 =====
file_client_args = dict(io_backend='disk')

# ===== SlowFast 官方的 4x16x1 采样设置（Decord + NCTHW）=====
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=10, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# ===== DataLoade =====
dataset_type = 'VideoDataset'
train_dataloader = dict(
    batch_size=14,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
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
        data_prefix=dict(video=data_root_val),
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
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True)
)

# ===== 评估与训练日程 =====
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=256, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=34, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=256, eta_min=0, by_epoch=True, begin=0, end=256)
]

# default_hooks = dict(
#     checkpoint=dict(interval=4, max_keep_ckpts=3),
#     logger=dict(interval=100)
# )

# ===== W&B 可视化配置 =====
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mmaction2',
            name='slowfast_devisignD_r50_256e',
            entity='soyorin',
            tags=['slowfast','DEVISIGN-D']
        ),
        define_metric_cfg=[
            dict(name='iter'),   # 记录迭代计数（供 train/* 作为 x 轴）
            dict(name='epoch'),  # 记录轮次计数（供 val/* 作为 x 轴）
            dict(name='train/*', step_metric='iter'),
            dict(name='val/*',   step_metric='epoch'),
            dict(name='val/top1_acc', summary='max'),
        ],
        commit=True
    )
]

visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=1,                # 每 1 个迭代记录一次
        log_metric_by_epoch=True   # 验证集按 epoch 作为步轴
    ),
    checkpoint=dict(type='CheckpointHook', interval=4, max_keep_ckpts=3, save_best='auto')
)

log_processor = dict(
    type='LogProcessor',
    window_size=1,
    by_epoch=True,
    log_with_hierarchy=True
)
