checkpoint_config = dict(interval=1, by_epoch=True, out_dir='/root/output/szb/checkpoints')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
resume_from = '/root/output/szb/checkpoints/latest.pth'
workflow = [('train', 1)]