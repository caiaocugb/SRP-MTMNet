import os.path as osp

import mmcv
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations
from mmdet.datasets.pipelines.loading import LoadPanopticAnnotations

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadMultiModalImageFromFile:
    """Load aligned optical and InSAR RGB images from one dataset record."""

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        prefix = results.get('img_prefix')
        opt_relative = results['img_info']['filename_opt']
        insar_relative = results['img_info']['filename_insar']
        opt_filename = (osp.join(prefix, opt_relative)
                        if prefix else opt_relative)
        insar_filename = (osp.join(prefix, insar_relative)
                          if prefix else insar_relative)

        img_opt = mmcv.imfrombytes(
            self.file_client.get(opt_filename), flag=self.color_type)
        img_insar = mmcv.imfrombytes(
            self.file_client.get(insar_filename), flag=self.color_type)
        if img_opt is None or img_insar is None:
            raise FileNotFoundError(
                f'Unable to load paired images: {opt_filename}, '
                f'{insar_filename}')
        if img_opt.shape != img_insar.shape:
            raise ValueError(
                f'Paired image shapes differ: {img_opt.shape} vs '
                f'{img_insar.shape}')

        if self.to_float32:
            img_opt = img_opt.astype(np.float32)
            img_insar = img_insar.astype(np.float32)

        results.update(
            filename=opt_filename,
            ori_filename=opt_relative,
            filename_opt=opt_filename,
            filename_insar=insar_filename,
            ori_filename_opt=opt_relative,
            ori_filename_insar=insar_relative,
            img=img_opt.copy(),
            img_opt=img_opt,
            img_insar=img_insar,
            img_shape=img_opt.shape,
            ori_shape=img_opt.shape,
            img_fields=['img', 'img_opt', 'img_insar'],
        )
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f"color_type='{self.color_type}')")


@PIPELINES.register_module()
class LoadSceneGraphAnnotations(LoadAnnotations):
    def __init__(
            self,
            with_bbox=True,
            with_label=True,
            with_mask=False,
            with_seg=False,
            poly2mask=True,
            file_client_args=dict(backend='disk'),
            # New args
            with_rel=False,
    ):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            file_client_args=dict(backend='disk'),
        )
        self.with_rel = with_rel

    def _load_rels(self, results):
        ann_info = results['ann_info']
        results['gt_rels'] = ann_info['rels']
        results['gt_relmaps'] = ann_info['rel_maps']

        assert 'rel_fields' in results

        results['rel_fields'] += ['gt_rels', 'gt_relmaps']
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_rel:
            results = self._load_rels(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()

        repr_str += f', with_rel={self.with_rel})'

        return repr_str


@PIPELINES.register_module()
class LoadPanopticSceneGraphAnnotations(LoadPanopticAnnotations):
    def __init__(
            self,
            with_bbox=True,
            with_label=True,
            with_mask=True,
            with_seg=True,
            file_client_args=dict(backend='disk'),
            # New args
            with_rel=False,
    ):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            file_client_args=dict(backend='disk'),
        )
        self.with_rel = with_rel

    def _load_rels(self, results):
        ann_info = results['ann_info']
        results['gt_rels'] = ann_info['rels']
        results['gt_relmaps'] = ann_info['rel_maps']

        assert 'rel_fields' in results

        results['rel_fields'] += ['gt_rels', 'gt_relmaps']
        return results

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(img_bytes,
                                   flag='color',
                                   channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # # The legal thing masks
            # if mask_info.get('is_thing'):
            #     gt_masks.append(mask.astype(np.uint8))
            gt_masks.append(mask.astype(np.uint8))  # get all masks

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            # print('origin_size')
            # print(gt_masks)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_rel:
            results = self._load_rels(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()

        repr_str += f', with_rel={self.with_rel})'

        return repr_str
