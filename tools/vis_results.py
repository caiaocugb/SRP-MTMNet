"""Visualize saved PSGFormer predictions on the optical modality."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    try:
        valid_thread_count = int(os.environ.get(thread_variable, '1')) >= 1
    except ValueError:
        valid_thread_count = False
    if not valid_thread_count:
        os.environ[thread_variable] = '1'

import mmcv
from mmcv import Config, DictAction
from mmcv.utils import import_modules_from_strings

from openpsg.datasets import build_dataset
from openpsg.utils.utils import show_result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='model config file')
    parser.add_argument('predictions', help='pickle produced by tools/test.py')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument(
        '--indices', nargs='+', type=int, help='sample indices to render')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    mmcv.check_file_exist(args.predictions)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.predictions)
    if len(outputs) != len(dataset):
        raise ValueError(
            f'Prediction count ({len(outputs)}) differs from dataset size '
            f'({len(dataset)})')

    indices = args.indices if args.indices is not None else range(len(dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        sample = dataset[index]
        image_path = sample['img_metas'][0].data['filename_opt']
        output_path = args.output_dir / f'{index:04d}_opt.png'
        show_result(
            image_path,
            outputs[index],
            is_one_stage=True,
            num_rel=args.topk,
            out_file=str(output_path),
        )
        print(output_path)


if __name__ == '__main__':
    main()
