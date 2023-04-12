_base_ = [
    '../_base_/datasets/ssdd.py', '../_base_/schedules/schedule_200e.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'  # 旋转定义方式

# model settings
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,    # 主干网络的深度
        num_stages=4,   # 主干网络阶段(stages)的数目
        out_indices=(0, 1, 2, 3),   # 每个阶段产生的特征图输出的索引
        frozen_stages=1,    # 第一个阶段的权重被冻结
        zero_init_residual=False,    # 是否对残差块(resblocks)中的最后一个归一化层使用零初始化(zero init)让它们表现为自身
        norm_cfg=dict(  # 归一化层(norm layer)的配置项
            type='BN',  # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True),    # 是否训练归一化里的 gamma 和 beta
        norm_eval=True, # 是否冻结 BN 里的统计项
        style='pytorch',     # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # 加载通过 ImageNet 预训练的模型
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,
        start_level=1,  # 用于构建特征金字塔的主干网络起始输入层索引值
        add_extra_convs='on_output',  # use P5  # 决定是否在原始特征图之上添加卷积层
        num_outs=5, # 决定输出多少个尺度的特征图(scales)
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=1,  # 分类的类别数量
        in_channels=256,    # bbox head 输入通道数
        stacked_convs=4,    # head 卷积层的层数
        feat_channels=256,  # head 卷积层的特征通道
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(    # 图像归一化配置，用来归一化输入的图像
    mean=[123.675, 116.28, 103.53],    # 图像归一化配置，用来归一化输入的图像
    std=[58.395, 57.12, 57.375], to_rgb=True)   # 预训练里用于预训练主干网络模型的标准差
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'), # 第 1 个流程，从文件路径里加载图像
    dict(type='LoadAnnotations', with_bbox=True),   # 第 2 个流程，对于当前图像，加载它的注释信息
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
