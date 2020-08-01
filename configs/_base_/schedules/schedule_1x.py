# optimizer
optimizer_gen = dict(
    type='Adam', 
    lr=0.001, 
    betas=(0,0.99), 
    eps=1e-8, 
    weight_decay=0.,
    paramwise_cfg=dict(
        custom_keys={
            'mapping_': dict(
                lr_mult=0.01
            )
        }
    ))
optimizer_disc = dict(type='Adam', lr=0.001, betas=(0,0.99), eps=1e-8, weight_decay=0.)

stage_epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]
fade_in_percentages = [50, 50, 50, 50, 50, 50, 50, 50, 50]