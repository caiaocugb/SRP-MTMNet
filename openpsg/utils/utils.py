"""Visualization utilities for SRP-MTMNet predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import Visualizer


CLASSES = ['None-SDZ', 'road', 'river', 'building', 'SDZ', 'background']
PREDICATES = ['connect', 'contain', 'adjacent_to']


def get_colormap(num_colors):
    """Return ``num_colors`` RGB colors in Detectron2 format."""

    return (np.resize(colormap(), (num_colors, 3)) / 255.0).tolist()


def _resize_masks(masks, width, height):
    resized = []
    for mask in np.asarray(masks):
        resized_mask = mmcv.imresize(
            mask.astype(np.uint8), (width, height), interpolation='nearest')
        resized.append(resized_mask.astype(bool))
    return resized


def _top_relationships(result, top_k):
    rel_dists = np.asarray(result.rel_dists)
    rel_pairs = np.asarray(result.rel_pair_idxes)
    if rel_dists.size == 0 or rel_pairs.size == 0:
        return []

    foreground = rel_dists[:, 1:]
    scores = foreground.max(axis=1)
    order = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    predicate_ids = foreground[order].argmax(axis=1)
    labels = np.asarray(result.labels).reshape(-1)

    relationships = []
    for prediction_index, predicate_id in zip(order, predicate_ids):
        subject_index, object_index = rel_pairs[prediction_index]
        subject_label = CLASSES[int(labels[subject_index]) - 1]
        object_label = CLASSES[int(labels[object_index]) - 1]
        relationships.append(
            (subject_label, PREDICATES[int(predicate_id)], object_label,
             float(scores[prediction_index])))
    return relationships


def show_result(
    img,
    result,
    is_one_stage=True,
    num_rel=20,
    show=False,
    out_dir=None,
    out_file=None,
):
    """Render entity masks and the highest-scoring relationship triplets."""

    del is_one_stage, out_dir
    image_bgr = mmcv.imread(img)
    image_rgb = mmcv.bgr2rgb(image_bgr)
    height, width = image_rgb.shape[:2]

    masks = _resize_masks(result.masks, width, height)
    labels = [CLASSES[int(label) - 1] for label in result.labels]
    colors = get_colormap(len(masks))

    visualizer = Visualizer(image_rgb)
    visualizer.overlay_instances(
        labels=labels,
        masks=masks,
        assigned_colors=colors,
    )
    overlay = visualizer.get_output().get_image()
    relationships = _top_relationships(result, num_rel)

    figure_height = max(4.5, 0.32 * max(len(relationships), 1))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, figure_height),
        gridspec_kw={'width_ratios': [2.2, 1.0]},
    )
    axes[0].imshow(overlay)
    axes[0].axis('off')
    axes[0].set_title('Predicted entities')
    axes[1].axis('off')
    axes[1].set_title(f'Top-{len(relationships)} relationships')

    if relationships:
        lines = [
            f'{index + 1:02d}. {subject} - {predicate} - {object} '
            f'({score:.3f})'
            for index, (subject, predicate, object, score) in
            enumerate(relationships)
        ]
        axes[1].text(
            0.0,
            1.0,
            '\n'.join(lines),
            transform=axes[1].transAxes,
            va='top',
            ha='left',
            fontsize=9,
            family='monospace',
        )
    else:
        axes[1].text(0.0, 1.0, 'No foreground relationships', va='top')

    fig.tight_layout()
    if out_file is not None:
        output_path = Path(out_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return overlay
