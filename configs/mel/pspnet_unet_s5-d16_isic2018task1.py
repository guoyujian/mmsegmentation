
_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', 
    '../_base_/datasets/mel.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]



model = dict(
    decode_head=dict(
        # out_channels=1,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        # out_channels = 1,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
)


# log_config = dict(
#     interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])



# workflow = [('train', 1)]
# cudnn_benchmark = True
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=4000)
# checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=100, metric='mDice', pre_eval=True)
