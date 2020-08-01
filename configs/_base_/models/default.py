model = dict(
    type='StyleGAN',
    resolution=1024,
    latent_channels=512,
    pretrained=None,
    generator=dict(
        type='StyleGANGen',
        mapping=dict(
            type='StyleMapping',
            in_channels=512,
            out_channels=512,
            num_layers=8,
            broadcast_num=18,
            activation=dict(
                type='LeakyReLU',
                alpha=0.2
            )
        ),
        synthesis=dict(
            type='StyleSynthesis',
            in_channels=512,
            style_channels=512,
            activation=dict(
                type='LeakyReLU',
                alpha=0.2
            )
        )
    ),
    discriminator=dict(
        type='ProGANDisc',
        max_channels=512,
        activation=dict(
            type='LeakyReLU',
            alpha=0.2
        )
    ),
    loss_gen=dict(
        type='ProGANLossGen',
    ),
    loss_disc=dict(
        type='ProGANLossDisc',
    )
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)