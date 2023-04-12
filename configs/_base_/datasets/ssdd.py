# dataset settings
dataset_type = 'SARDataset'
data_root = 'data/ssdd/'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/all/labelTxt/',
        data_prefix=dict(img_path='test/all/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='test/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/dota/Task1')




# # dataset settings
# dataset_type = 'mmdet.CocoDataset'
# data_root = 'data/ssdd/'
# backend_args = None

# train_pipeline = [
#     dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
#     dict(
#         type='mmdet.LoadAnnotations',
#         with_bbox=True,
#         with_mask=True,
#         poly2mask=False),
#     dict(type='ConvertMask2BoxType', box_type='rbox'),
#     dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
#     dict(
#         type='mmdet.RandomFlip',
#         prob=0.75,
#         direction=['horizontal', 'vertical', 'diagonal']),
#     dict(type='mmdet.PackDetInputs')
# ]
# val_pipeline = [
#     dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
#     dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
#     # avoid bboxes being resized
#     dict(
#         type='mmdet.LoadAnnotations',
#         with_bbox=True,
#         with_mask=True,
#         poly2mask=False),
#     dict(type='ConvertMask2BoxType', box_type='qbox'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'instances'))
# ]
# test_pipeline = [
#     dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
#     dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# metainfo = dict(classes=('ship', ))

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=None,
#     dataset=dict(
#         type=dataset_type,
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='train/train.json',
#         data_prefix=dict(img='train/images/'),
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline=train_pipeline,
#         backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='test/all/test.json',
#         data_prefix=dict(img='test/all/images/'),
#         test_mode=True,
#         pipeline=val_pipeline,
#         backend_args=backend_args))
# test_dataloader = val_dataloader

# val_evaluator = dict(type='RotatedCocoMetric', metric='bbox',  backend_args=backend_args)

# test_evaluator = val_evaluator
