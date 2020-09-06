dataset_type = 'ImgFolderDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromPath'),
    dict(type='EasyResize', img_scale=(1024)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(128),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_folder='/root/data1024x1024',
        pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     img_folder=data_root + 'val',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     img_folder=data_root + 'test',
    #     pipeline=test_pipeline))
)
# evaluation = dict(interval=1, metric='bbox')