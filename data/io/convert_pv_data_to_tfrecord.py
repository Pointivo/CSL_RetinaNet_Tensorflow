# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import sys
from pathlib import Path
from typing import List, Dict

import PIL.Image as Image

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import cv2
from help_utils.tools import *
import json

NAME_TO_LABEL_MAP = {
    'back_ground': 0,
    'penetration': 1
}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_image_name_from_od_json_file(od_file_path: Path):
    with od_file_path.open() as f:
        od_data = json.load(f)
    return od_data['imageName']


def _get_bounding_boxes_from_od_json_file(od_file_path: Path, class_name_to_label_map: Dict[str, int]) -> np.ndarray:
    with od_file_path.open() as f:
        od_data = json.load(f)
    bounding_boxes = []
    for bbox in od_data['boundingBoxes']:
        x1, x2, x3, x4 = bbox['x1'], bbox['x2'], bbox['x3'], bbox['x4']
        y1, y2, y3, y4 = bbox['y1'], bbox['y2'], bbox['y3'], bbox['y4']
        class_label = class_name_to_label_map[bbox['classLabel']]
        bbox = [x1, y1, x2, y2, x3, y3, x4, y4, class_label]
        bounding_boxes.append(bbox)
    return np.array(bounding_boxes, dtype=np.int32)


def _pad_image_with_zeros(image_np: np.ndarray) -> np.ndarray:
    # this method pads image with zeros appropriately to make it square. Note that this method appends on the
    # bottom and right side of the image so there is no need to adjust the bounding box coordinates
    assert len(image_np.shape) == 3 and image_np.shape[2] == 3, 'image shape not supported. {}'.format(image_np.shape)
    image_h, image_w, _ = image_np.shape
    if image_h == image_w:
        return image_np
    elif image_h < image_w:
        padded_image = np.zeros([image_w, image_w, 3])
        padded_image[:image_h, :image_w] = image_np
        return padded_image
    else:
        padded_image = np.zeros([image_h, image_h, 3])
        padded_image[:image_h, :image_w] = image_np
        return padded_image


def _scale_bbox(bbox: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
    x1, y1, x2, y2, x3, y3, x4, y4, class_label = bbox
    x1, x2, x3, x4 = x1 * x_scale, x2 * x_scale, x3 * x_scale, x4 * x_scale
    y1, y2, y3, y4 = y1 * y_scale, y2 * y_scale, y3 * y_scale, y4 * y_scale
    return np.array([x1, y1, x2, y2, x3, y3, x4, y4, class_label], dtype=np.int32)


def _scale_bboxes(bboxes: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
    rescaled_bboxes = []
    for bbox in bboxes:
        rescaled_bbox = _scale_bbox(bbox=bbox, x_scale=x_scale, y_scale=y_scale)
        rescaled_bboxes.append(rescaled_bbox)
    return np.array(rescaled_bboxes, dtype=np.int32)


def _check_if_box_is_smaller_than_thresh_after_scaling(bbox: np.ndarray, x_scale: int, y_scale: int,
                                                       box_dim_thresh: float) -> bool:
    box_rescaled = _scale_bbox(bbox=bbox, x_scale=x_scale, y_scale=y_scale)[:-1].reshape(4, 2)
    rect = cv2.minAreaRect(np.int0(box_rescaled))
    (_, _), (box_w, box_h), _ = rect
    if box_w < box_dim_thresh or box_h < box_dim_thresh:
        return True
    return False


def read_od_json_gtbox_and_label(od_path, img_w, img_h):
    with open(od_path) as f:
        data = json.load(f)
    img_width = data['height']
    img_height = data['width']
    box_list = []
    new_w = new_h = cfgs.IMG_SHORT_SIDE_LEN
    bounding_boxes = data['boundingBoxes']
    for bbox in bounding_boxes:
        tmp_box = []
        x1 = bbox['x1']
        y1 = bbox['y1']
        x2 = bbox['x2']
        y2 = bbox['y2']
        x3 = bbox['x3']
        y3 = bbox['y3']
        x4 = bbox['x4']
        y4 = bbox['y4']
        x1, x2, x3, x4 = x1 * new_w // img_w, x2 * new_w // img_w, x3 * new_w // img_w, x4 * new_w // img_w
        y1, y2, y3, y4 = y1 * new_h // img_h, y2 * new_h // img_h, y3 * new_h // img_h, y4 * new_h // img_h
        box = np.array((x1, y1, x2, y2, x3, y3, x4, y4))

        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(np.int0(box))

        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        if w < 1 or h < 1:
            print('Ignoring BBox with height/width less 1 pixel')
            continue

        tmp_box.append(bbox['x1'])
        tmp_box.append(bbox['y1'])
        tmp_box.append(bbox['x2'])
        tmp_box.append(bbox['y2'])
        tmp_box.append(bbox['x3'])
        tmp_box.append(bbox['y3'])
        tmp_box.append(bbox['x4'])
        tmp_box.append(bbox['y4'])
        tmp_box.append(NAME_TO_LABEL_MAP[bbox['classLabel']])
        box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def _remove_bbox_smaller_than_thresh_after_resizing(bboxes: np.ndarray, image_size: int,
                                                    min_box_dim_thresh: float = 5) -> np.ndarray:
    if isinstance(cfgs.IMG_SHORT_SIDE_LEN, (list, tuple)):
        new_image_size = min(cfgs.IMG_SHORT_SIDE_LEN)
    else:
        new_image_size = cfgs.IMG_SHORT_SIDE_LEN
    scale = new_image_size / image_size
    bboxes_to_keep = []
    for bbox in bboxes:
        if not _check_if_box_is_smaller_than_thresh_after_scaling(box=bbox, x_scale=scale, y_scale=scale,
                                                                  box_dim_thresh=min_box_dim_thresh):
            bboxes_to_keep.append(bbox)
    return np.array(bboxes_to_keep)


def convert_pv_data_to_tfrecord(pv_data_dir: Path, dataset: str, save_name: str,
                                class_name_to_label_map: Dict[str, int], max_image_size: int = 1500,
                                min_box_dim_thresh: float = 3):
    save_dir = '../tfrecord/'
    tfrecord_file_path = os.path.join(save_dir, dataset + '_' + save_name + '.tfrecord')
    mkdir(save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=tfrecord_file_path)

    od_json_files = list(pv_data_dir.glob('*.ground_truth.od.json'))
    total_od_json_files = len(od_json_files)
    for count, od_file_path in enumerate(od_json_files):
        image_name = _get_image_name_from_od_json_file(od_file_path=od_file_path)
        assert '.jpg' in str(image_name), 'Only .jpg images are currently supported.'
        image_path = pv_data_dir / image_name

        if not image_path.exists():
            raise AssertionError('{} should exist in the dataset!'.format(image_path))

        image = Image.open(str(image_path))
        image = np.asarray(image)
        image = _pad_image_with_zeros(image_np=image)
        image = image.astype(np.uint8)
        assert image.shape[0] == image.shape[1] and image.shape[2] == 3 and len(image.shape) == 3, \
            'image shape not correct: {}'.format(image.shape)
        image_size = image.shape[0]

        bounding_boxes = _get_bounding_boxes_from_od_json_file(od_file_path=od_file_path,
                                                               class_name_to_label_map=class_name_to_label_map)
        bounding_boxes = _remove_bbox_smaller_than_thresh_after_resizing(bboxes=bounding_boxes, image_size=image_size,
                                                                         min_box_dim_thresh=min_box_dim_thresh)

        if image_size > max_image_size:
            image = cv2.resize(image, dsize=(max_image_size, max_image_size), interpolation=cv2.INTER_AREA)
            image = image.astype(np.uint8)
            assert image.shape[0] == image.shape[1] and image.shape[2] == 3 and len(image.shape) == 3, \
                'image shape not correct: {}'.format(image.shape)
            scale_factor = max_image_size / image_size
            image_size = image.shape[0]
            bounding_boxes = _scale_bboxes(bboxes=bounding_boxes, x_scale=scale_factor, y_scale=scale_factor)

        if len(bounding_boxes) > 0:
            success, image_jpg_encoded = cv2.imencode('.jpg', image[:, :, ::-1])  # convert RGB to BGR before using cv2
            assert success is True, 'JPG encoding failed for {}'.format(image_name)
            feature = tf.train.Features(feature={
                'img_name': _bytes_feature(image_name.encode()),
                'img_height': _int64_feature(int(image_size)),
                'img_width': _int64_feature(int(image_size)),
                'img': _bytes_feature(image_jpg_encoded.tostring()),
                'gtboxes_and_label': _bytes_feature(bounding_boxes.tostring()),
                'num_objects': _int64_feature(len(bounding_boxes))})
            example = tf.train.Example(features=feature)
            writer.write(example.SerializeToString())
        else:
            print('\nSkipping image {} as no bboxes large enough.'.format(image_name))
        view_bar('Conversion progress', count + 1, total_od_json_files)

    print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    pv_data_dir = Path('path-to-pv-dataset')
    class_name_to_label_map = {'back_ground': 0, 'penetration': 1}
    convert_pv_data_to_tfrecord(pv_data_dir=pv_data_dir, dataset='PENETRATION', save_name='train', max_image_size=1500,
                                min_box_dim_thresh=3, class_name_to_label_map=class_name_to_label_map)
