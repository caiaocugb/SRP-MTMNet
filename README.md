# SRP-MTMNet

SRP-MTMNet is a dual-stream multimodal PSGFormer for joint panoptic entity
segmentation and object-level spatial-relationship prediction. It accepts a
spatially aligned optical RGB image and an RGB MT-InSAR velocity image, extracts
features with two ResNet-50 backbones, and fuses each pair of aligned feature
maps with concatenation followed by a 1x1 convolution, batch normalization, and
ReLU. The fused features are passed to the PSGFormer head to predict entity
masks and relationship triplets.

## Classes

Entity classes:

1. None-SDZ
2. road
3. river
4. building
5. SDZ（Significant deformation zone）
6. background

Relationship classes:

1. connect
2. contain
3. adjacent_to

## Repository layout

```text
configs/psgformer/     Model and training configuration
data/                  Four paired example samples and panoptic annotations
openpsg/               Dataset, model, loss, evaluation, and visualization code
scripts/               Convenience commands
tools/train.py         Training entry point
tools/test.py          Evaluation and prediction entry point
tools/vis_results.py   Prediction visualization
```

The bundled example dataset contains two training samples and two test samples.
Users can refer to this example and the COCO dataset provided in the official OpenPSG repository to prepare their own datasets.

## Environment

The original experiments used Python 3.8, PyTorch 1.7.0, CUDA 10.1, and two
NVIDIA RTX 2080 Ti GPUs. The OpenPSG dependency stack is version-sensitive.

```bash
conda env create -f environment.yml
conda activate srp-mtmnet

pip install mmcv-full==1.4.3 \
  -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
mim install mmdet==2.20.0
pip install git+https://github.com/cocodataset/panopticapi.git
pip install detectron2==0.5 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
pip install -e .
```

## Example data

The default config reads `data/SRP_MTMNet_TGRA.json`. Image paths inside the
JSON are relative to `data/`.

```text
data/
├── SRP_MTMNet_TGRA.json
├── train/
│   ├── opt/
│   └── insar/
├── test/
│   ├── opt/
│   └── insar/
├── panoptic_train/
└── panoptic_test/
```

Optical and InSAR images in each pair must have identical height and width.
Panoptic PNG segment IDs follow the COCO `rgb2id` encoding.

##Notice!
The functions in the mmdet library need to be modified to adapt to dual-branch data input and feature extraction.

## Train

```bash
python tools/train.py configs/psgformer/psgformer_r50_psg.py \
  --work-dir work_dirs/psgformer_r50_psg \
```

## Evaluate and export predictions

```bash
python tools/test.py \
  configs/psgformer/psgformer_r50_psg.py \
  work_dirs/psgformer_r50_psg/latest.pth \
  --out work_dirs/psgformer_r50_psg/test_results.pkl \
  --work-dir work_dirs/psgformer_r50_psg/evaluation \
```

## Visualize predictions

```bash
python tools/vis_results.py \
  configs/psgformer/psgformer_r50_psg.py \
  work_dirs/psgformer_r50_psg/test_results.pkl \
  outputs/visualizations --topk 50
```
