"""Validate paired images and panoptic annotations in an SRP-MTMNet dataset."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'annotation',
        nargs='?',
        type=Path,
        default=Path('data/SRP_MTMNet_TGRA.json'),
    )
    return parser.parse_args()


def rgb2id(rgb):
    rgb = rgb.astype(np.int64)
    return rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]


def main():
    annotation_path = parse_args().annotation.resolve()
    data_root = annotation_path.parent
    dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
    records = dataset['data']
    test_ids = set(dataset['test_image_ids'])
    class_count = len(dataset['thing_classes']) + len(dataset['stuff_classes'])
    predicate_count = len(dataset['predicate_classes'])

    errors = []
    shared_id_counts = {}
    seen_ids = set()

    for record in records:
        image_id = record['image_id']
        if image_id in seen_ids:
            errors.append(f'{image_id}: duplicate image_id')
        seen_ids.add(image_id)

        paths = {
            'optical': data_root / record['file_name_opt'],
            'insar': data_root / record['file_name_insar'],
            'panoptic': data_root / record['pan_seg_file_name'],
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            errors.append(f'{image_id}: missing files: {", ".join(missing)}')
            continue

        expected_size = (record['width'], record['height'])
        sizes = {name: Image.open(path).size for name, path in paths.items()}
        for name, size in sizes.items():
            if size != expected_size:
                errors.append(
                    f'{image_id}: {name} size {size} != {expected_size}'
                )

        is_test = image_id in test_ids
        expected_split = 'test/' if is_test else 'train/'
        if not record['file_name_opt'].startswith(expected_split):
            errors.append(f'{image_id}: optical path has the wrong split')
        if not record['file_name_insar'].startswith(expected_split):
            errors.append(f'{image_id}: InSAR path has the wrong split')

        segments = record['segments_info']
        annotations = record['annotations']
        if len(segments) != len(annotations):
            errors.append(
                f'{image_id}: {len(segments)} segments but '
                f'{len(annotations)} annotations'
            )

        for index, (segment, annotation) in enumerate(
                zip(segments, annotations)):
            if segment['category_id'] != annotation['category_id']:
                errors.append(f'{image_id}: category mismatch at entity {index}')
            if not 0 <= segment['category_id'] < class_count:
                errors.append(f'{image_id}: invalid category at entity {index}')

        entity_count = len(segments)
        for relation_index, relation in enumerate(record['relations']):
            if len(relation) != 3:
                errors.append(
                    f'{image_id}: malformed relation {relation_index}'
                )
                continue
            subject, object_, predicate = relation
            if not 0 <= subject < entity_count or not 0 <= object_ < entity_count:
                errors.append(
                    f'{image_id}: invalid entity index in relation '
                    f'{relation_index}'
                )
            if not 0 <= predicate < predicate_count:
                errors.append(
                    f'{image_id}: invalid predicate in relation '
                    f'{relation_index}'
                )

        panoptic_rgb = np.asarray(Image.open(paths['panoptic']).convert('RGB'))
        panoptic_ids = rgb2id(panoptic_rgb)
        present_ids, pixel_counts = np.unique(panoptic_ids, return_counts=True)
        actual_area = dict(zip(present_ids.tolist(), pixel_counts.tolist()))
        declared_area = defaultdict(int)
        for segment in segments:
            declared_area[segment['id']] += segment['area']

        if set(actual_area) != set(declared_area):
            errors.append(f'{image_id}: panoptic and JSON segment IDs differ')
        for segment_id in set(actual_area) & set(declared_area):
            if actual_area[segment_id] != declared_area[segment_id]:
                errors.append(
                    f'{image_id}: area mismatch for segment {segment_id}'
                )

        repeated = sum(
            count - 1
            for count in Counter(s['id'] for s in segments).values()
            if count > 1
        )
        if repeated:
            shared_id_counts[image_id] = repeated

    if not test_ids <= seen_ids:
        errors.append('test_image_ids contains IDs not present in data')

    if errors:
        print('Validation failed:')
        for error in errors:
            print(f'  - {error}')
        raise SystemExit(1)

    train_count = len(records) - len(test_ids)
    print(
        f'Validation passed: {len(records)} records '
        f'({train_count} train, {len(test_ids)} test).'
    )
    if shared_id_counts:
        details = ', '.join(
            f'{image_id}={count}'
            for image_id, count in sorted(shared_id_counts.items())
        )
        print(f'Source-format shared segment-ID entries: {details}')


if __name__ == '__main__':
    main()
