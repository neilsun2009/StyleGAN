model = dict(
    type='StyleGAN',
    resolution=1024,
    latent_channels=512,
    pretrained=None,
    generator=dict(
        type='StyleGANGen',
        # mapping=dict(
        #     type='StyleMapping',
        #     in_channels=512,
        #     out_channels=512,
        #     num_layers=8,
        #     activation=dict(
        #         type='LeakyReLU',
        #         alpha=0.2
        #     ),
        # ),
        mapping=dict(
            type='RefMapping',
            latent_size=512,
            dlatent_size=512,
            mapping_layers=8,
            dlatent_broadcast=18
        ),
        # synthesis=dict(
        #     type='StyleSynthesis',
        #     in_channels=512,
        #     style_channels=512,
        #     activation=dict(
        #         type='LeakyReLU',
        #         alpha=0.2
        #     )
        # )
        synthesis=dict(
            type='RefSynthesis',
            dlatent_size=512,
            blur_filter = [1, 2, 1]
        )
    ),
    discriminator=dict(
        # type='ProGANDisc',
        # max_channels=512,
        # activation=dict(
        #     type='LeakyReLU',
        #     alpha=0.2
        # )
        type='RefDisc',
        blur_filter = [1, 2, 1]
    ),
    loss_gen=dict(
        type='LogisticLossGen',
    ),
    loss_disc=dict(
        type='LogisticLossDisc',
    )
)
