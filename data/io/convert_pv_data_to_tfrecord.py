# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import sys

import PIL.Image as Image

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import glob
import io
import cv2
from libs.label_name_dict.label_dict import *
from help_utils.tools import *
import json

NAME_LABEL_MAP = {
    'back_ground': 0,
    'penetration': 1
}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
        tmp_box.append(NAME_LABEL_MAP[bbox['classLabel']])
        box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def convert_pv_data_to_tfrecord():
    data_dir = '/mnt/1tbssd/adnan/python-microservices/image-recognition/image_recognition/tmp/penetrations_05192020_train'
    # data_dir = '../penetrations_05192020_patchwise_val'
    od_path = data_dir
    image_path = data_dir
    save_name = 'train'
    save_dir = '../tfrecord/' + save_name
    dataset = 'PENETRATION'
    img_format = '.jpg'
    save_path = os.path.join(save_dir, dataset + '_' + save_name + '.tfrecord')
    mkdir(save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    for count, od in enumerate(glob.glob(od_path + '/*_oriented.ground_truth.od.json')):

        img_name = od.split('/')[-1].split('.')[0][:-9] + img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        image = np.asarray(image)

        _, _, gtbox_label = read_od_json_gtbox_and_label(od, image.shape[0], image.shape[1])

        feature = tf.train.Features(feature={
            'img_name': _bytes_feature(img_name.encode()),
            'img_height': _int64_feature(int(image.shape[0])),
            'img_width': _int64_feature(int(image.shape[1])),
            'img': _bytes_feature(encoded_jpg),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(od_path + '/*_oriented.ground_truth.od.json')))

    print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    convert_pv_data_to_tfrecord()
