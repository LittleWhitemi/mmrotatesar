# evaluation
evaluation = dict(  # evaluation hook 的配置
    interval=1, # 验证的间隔
    metric='mAP')   # 验证期间使用的指标
# optimizer
optimizer = dict(   # 用于构建优化器的配置文件
    type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))   # optimizer hook 的配置文件
# learning policy
lr_config = dict(   # 学习率调整配置，用于注册 LrUpdater hook
    policy='step',  # 调度流程(scheduler)的策略
    warmup='linear',    # 预热(warmup)策略，也支持 `exp` 和 `constant`
    warmup_iters=500,   # 预热的迭代次数
    warmup_ratio=1.0 / 3,   # 用于预热的起始学习率的比率
    step=[32, 64, 96, 128, 160, 180])   # 衰减学习率的起止回合数
runner = dict(type='EpochBasedRunner',   # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)
              max_epochs=200)    # runner 总回合(epoch)数， 对于 IterBasedRunner 使用 `max_iters`
checkpoint_config = dict(   # checkpoint hook 的配置文件
    interval=20) # 保存的间隔是 12
