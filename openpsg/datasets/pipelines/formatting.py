import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class RelsFormatBundle(DefaultFormatBundle):
    """Format paired images and relationship annotations for training."""

    def __call__(self, results):
        alias_added = 'img' not in results and 'img_opt' in results
        if alias_added:
            results['img'] = results['img_opt']
        results = super().__call__(results)
        if alias_added:
            results.pop('img', None)

        for key in ('img_opt', 'img_insar'):
            if key not in results or isinstance(results[key], DC):
                continue
            image = results[key]
            if image.dtype == np.uint8 and self.img_to_float:
                image = image.astype(np.float32)
            if image.ndim < 3:
                image = np.expand_dims(image, -1)
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            results[key] = DC(
                to_tensor(image), stack=True, padding_value=self.pad_val['img'])

        if 'gt_rels' in results and not isinstance(results['gt_rels'], DC):
            results['gt_rels'] = DC(to_tensor(results['gt_rels']))

        return results
