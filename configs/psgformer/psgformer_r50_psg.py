_base_ = ['./psgformer_r50.py', '../_base_/custom_runtime.py']

find_unused_parameters = True

custom_imports = dict(
    imports=[
        'openpsg.models.frameworks.psgtr',
        'openpsg.models.frameworks.dual_transformer',
        'openpsg.models.losses.seg_losses',
        'openpsg.models.relation_heads.psgformer_head',
        'openpsg.models.relation_heads.approaches.matcher',
        'openpsg.datasets',
        'openpsg.datasets.pipelines.loading',
        'openpsg.datasets.pipelines.rel_randomcrop',
        'openpsg.utils',
    ],
    allow_failed_imports=False,
)

dataset_type = 'PanopticSceneGraphDataset'
#data_root = 'data'
#ann_file = f'{data_root}/SRP_MTMNet_TGRA.json'
data_root = '/root/autodl-tmp/OpenPSG-main/data/psg_tgra_train'
ann_file = f'{data_root}/psg_tgra_all.json'

object_classes = [
    'None-SDZ', 'road', 'river', 'building', 'SDZ', 'background'
]
predicate_classes = ['connect', 'contain', 'adjacent_to']

model = dict(
    bbox_head=dict(
        num_classes=len(object_classes),
        num_relations=len(predicate_classes),
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_obj_query=100,
        num_rel_query=100,
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

meta_keys = (
    'filename', 'filename_opt', 'filename_insar', 'ori_filename',
    'ori_filename_opt', 'ori_filename_insar', 'ori_shape', 'img_shape',
    'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'
)

train_pipeline = [
    dict(type='LoadMultiModalImageFromFile'),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333)
                    ],
                    multiscale_mode='value',
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True,
                ),
                dict(
                    type='RelRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False,
                ),
                dict(
                    type='Resize',
                    img_scale=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333)
                    ],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='RelsFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img_opt', 'img_insar', 'gt_bboxes', 'gt_labels', 'gt_rels',
            'gt_masks'
        ],
        meta_keys=meta_keys,
    ),
]

test_pipeline = [
    dict(type='LoadMultiModalImageFromFile'),
    dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img_opt', 'img_insar']),
            dict(type='Collect', keys=['img_opt', 'img_insar'],
                 meta_keys=meta_keys),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=data_root,
        seg_prefix=data_root,
        pipeline=train_pipeline,
        split='train',
        all_bboxes=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=data_root,
        seg_prefix=data_root,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=data_root,
        seg_prefix=data_root,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True,
    ),
)

evaluation = dict(
    interval=1,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg',
)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'transformer.encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'transformer.decoder1': dict(lr_mult=0.1, decay_mult=1.0),
            'obj_query_embed': dict(lr_mult=0.1, decay_mult=1.0),
            'input_proj': dict(lr_mult=0.1, decay_mult=1.0),
            'class_embed': dict(lr_mult=0.1, decay_mult=1.0),
            'box_embed': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_attention': dict(lr_mult=0.1, decay_mult=1.0),
            'mask_head': dict(lr_mult=0.1, decay_mult=1.0),
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=10)
runner = dict(type='EpochBasedRunner', max_epochs=30)

work_dir = './work_dirs/psgformer_r50_psg'
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
#https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fwork%5Fdirs%2Fpsgformer%5Fr50&ga=1
#For the first training session, you can load the official OpenPSG weights.
#load_from = 'epoch_60.pth'
load_from = None
