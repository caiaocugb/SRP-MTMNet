# Copyright (c) OpenMMLab. All rights reserved.
"""Dual-stream PSGFormer detector used by SRP-MTMNet."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS, build_backbone, build_head
from mmdet.models.detectors.base import BaseDetector

from openpsg.models.relation_heads.approaches.relation_util import Result


class ConvFusion(nn.Module):
    """Fuse two aligned feature maps with a 1x1 convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, optical, insar):
        return self.fuse(torch.cat([optical, insar], dim=1))


def triplet2result(triplets, use_mask, eval_pan_rels=True):
    """Convert PSGFormer outputs to the common OpenPSG ``Result`` format."""

    if use_mask:
        (bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg,
         complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks,
         rels, pan_labels) = triplets

        if isinstance(bboxes, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            rel_pairs = rel_pairs.detach().cpu().numpy()
            complete_r_labels = complete_r_labels.detach().cpu().numpy()
            complete_r_dists = complete_r_dists.detach().cpu().numpy()
            r_labels = r_labels.detach().cpu().numpy()
            r_dists = r_dists.detach().cpu().numpy()
        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.detach().cpu().numpy()
            pan_rel_pairs = pan_rel_pairs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            pan_masks = pan_masks.detach().cpu().numpy()
            rels = rels.detach().cpu().numpy()
            pan_labels = pan_labels.detach().cpu().numpy()

        if eval_pan_rels:
            return Result(
                refine_bboxes=bboxes,
                labels=pan_labels + 1,
                formatted_masks=dict(pan_results=pan_seg),
                rel_pair_idxes=pan_rel_pairs,
                rel_dists=r_dists,
                rel_labels=r_labels,
                pan_results=pan_seg,
                masks=pan_masks,
                rels=rels,
            )
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=pan_seg),
            rel_pair_idxes=rel_pairs,
            rel_dists=complete_r_dists,
            rel_labels=complete_r_labels,
            pan_results=pan_seg,
            masks=masks,
        )

    bboxes, labels, rel_pairs, r_labels, r_dists = triplets
    return Result(
        refine_bboxes=bboxes.detach().cpu().numpy(),
        labels=labels.detach().cpu().numpy(),
        formatted_masks=dict(pan_results=None),
        rel_pair_idxes=rel_pairs.detach().cpu().numpy(),
        rel_dists=r_dists.detach().cpu().numpy(),
        rel_labels=r_labels.detach().cpu().numpy(),
        pan_results=None,
    )


@DETECTORS.register_module()
class PSGTr(BaseDetector):
    """PSGFormer with independent optical and InSAR ResNet backbones.

    The implementation works with an unmodified MMDetection 2.x installation.
    Aligned feature maps are concatenated and compressed with a 1x1
    convolution, batch normalization, and ReLU at each backbone level.
    """

    def __init__(
        self,
        backbone1,
        backbone2,
        bbox_head,
        fusion_channels=(256, 512, 1024, 2048),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if pretrained is not None:
            warnings.warn(
                "The detector-level pretrained argument is deprecated; use "
                "init_cfg in each backbone instead.",
                DeprecationWarning,
            )

        self.backbone1 = build_backbone(backbone1)
        self.backbone2 = build_backbone(backbone2)
        self.fusion_channels = tuple(fusion_channels)
        self.fusion_convs = nn.ModuleList([
            ConvFusion(2 * channels, channels)
            for channels in self.fusion_channels
        ])

        head_cfg = bbox_head.copy()
        head_in_channels = head_cfg.get("in_channels")
        if head_in_channels != self.fusion_channels[-1]:
            raise ValueError(
                f"bbox_head.in_channels={head_in_channels}, but the final "
                f"fusion level outputs {self.fusion_channels[-1]} channels"
            )
        expected_fpn_channels = tuple(reversed(self.fusion_channels[:-1]))
        head_fpn_channels = tuple(head_cfg.get("fpn_channels", ()))
        if head_fpn_channels != expected_fpn_channels:
            raise ValueError(
                f"bbox_head.fpn_channels={head_fpn_channels}, but the "
                f"fusion pyramid requires {expected_fpn_channels}"
            )
        head_cfg.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.bbox_head = build_head(head_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes

    @property
    def with_neck(self):
        return False

    def init_weights(self):
        super().init_weights()
        for fusion in self.fusion_convs:
            nn.init.kaiming_normal_(
                fusion.fuse[0].weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            nn.init.constant_(fusion.fuse[1].weight, 1)
            nn.init.constant_(fusion.fuse[1].bias, 0)

    def extract_feat(self, img):
        raise NotImplementedError(
            "PSGTr requires paired inputs; use extract_feat_opt and "
            "extract_feat_insar."
        )

    def extract_feat_opt(self, img_opt):
        return self.backbone1(img_opt)

    def extract_feat_insar(self, img_insar):
        return self.backbone2(img_insar)

    def _fuse_features(self, optical_features, insar_features):
        feature_count = len(optical_features)
        if feature_count != len(insar_features):
            raise ValueError("Optical and InSAR feature counts differ")
        if feature_count != len(self.fusion_convs):
            raise ValueError(
                f"Expected {len(self.fusion_convs)} feature levels, "
                f"but received {feature_count}"
            )
        fused_features = []
        for level, (fusion, expected_channels, optical, insar) in enumerate(
                zip(
                    self.fusion_convs,
                    self.fusion_channels,
                    optical_features,
                    insar_features,
                )):
            optical_channels = optical.shape[1]
            insar_channels = insar.shape[1]
            if (optical_channels != expected_channels
                    or insar_channels != expected_channels):
                raise ValueError(
                    f"Feature level {level} expects {expected_channels} "
                    f"channels per modality, but received "
                    f"{optical_channels} and {insar_channels}"
                )
            if optical.shape[-2:] != insar.shape[-2:]:
                raise ValueError(
                    f"Feature level {level} has different spatial shapes: "
                    f"{tuple(optical.shape[-2:])} vs "
                    f"{tuple(insar.shape[-2:])}"
                )
            fused_features.append(fusion(optical, insar))
        return tuple(fused_features)

    def forward_dummy(self, img_opt, img_insar):
        warnings.warn(
            "Multi-head attention FLOPs are not supported; do not report "
            "these estimates in a paper."
        )
        batch_size, _, height, width = img_opt.shape
        img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
            )
            for _ in range(batch_size)
        ]
        features = self._fuse_features(
            self.extract_feat_opt(img_opt),
            self.extract_feat_insar(img_insar),
        )
        return self.bbox_head(features, img_metas)

    @auto_fp16(apply_to=("img_opt", "img_insar"))
    def forward(
        self,
        img_opt,
        img_insar,
        img_metas,
        return_loss=True,
        **kwargs,
    ):
        if return_loss:
            return self.forward_train(
                img_opt=img_opt,
                img_insar=img_insar,
                img_metas=img_metas,
                **kwargs,
            )
        return self.forward_test(
            img_opt=img_opt,
            img_insar=img_insar,
            img_metas=img_metas,
            **kwargs,
        )

    def forward_train(
        self,
        img_opt,
        img_insar,
        img_metas,
        gt_rels,
        gt_bboxes,
        gt_labels,
        gt_masks,
        gt_bboxes_ignore=None,
    ):
        if img_opt.shape[-2:] != img_insar.shape[-2:]:
            raise ValueError(
                "Optical and InSAR batch dimensions differ: "
                f"{tuple(img_opt.shape[-2:])} vs "
                f"{tuple(img_insar.shape[-2:])}"
            )
        batch_input_shape = tuple(img_opt.shape[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

        features = self._fuse_features(
            self.extract_feat_opt(img_opt),
            self.extract_feat_insar(img_insar),
        )

        if self.bbox_head.use_mask:
            _, _, height, width = img_opt.shape
            resized_masks = []
            for masks in gt_masks:
                mask = torch.as_tensor(
                    masks.to_ndarray(), device=features[0].device
                )
                _, mask_height, mask_width = mask.shape
                padding = (0, width - mask_width, 0, height - mask_height)
                mask = F.interpolate(
                    F.pad(mask, padding).unsqueeze(1),
                    size=(height // 2, width // 2),
                    mode="nearest",
                ).squeeze(1)
                resized_masks.append(mask)
            gt_masks = resized_masks

        return self.bbox_head.forward_train(
            features,
            img_metas,
            gt_rels,
            gt_bboxes,
            gt_labels,
            gt_masks,
            gt_bboxes_ignore,
        )

    def forward_test(self, img_opt, img_insar, img_metas, **kwargs):
        if not isinstance(img_opt, list):
            img_opt = [img_opt]
        if not isinstance(img_insar, list):
            img_insar = [img_insar]
        if not isinstance(img_metas, list):
            img_metas = [img_metas]

        if not (len(img_opt) == len(img_insar) == len(img_metas)):
            raise ValueError("Optical, InSAR, and metadata augmentations differ")
        if len(img_opt) != 1:
            raise NotImplementedError("Test-time augmentation is not supported")

        for meta in img_metas[0]:
            meta["batch_input_shape"] = tuple(img_opt[0].size()[-2:])
        return self.simple_test(
            img_opt[0],
            img_insar[0],
            img_metas[0],
            rescale=kwargs.get("rescale", False),
        )

    def simple_test(self, img_opt, img_insar, img_metas, rescale=False):
        features = self._fuse_features(
            self.extract_feat_opt(img_opt),
            self.extract_feat_insar(img_insar),
        )
        triplets = self.bbox_head.simple_test(
            features, img_metas, rescale=rescale
        )
        return [
            triplet2result(output, self.bbox_head.use_mask)
            for output in triplets
        ]

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError("Test-time augmentation is not supported")
