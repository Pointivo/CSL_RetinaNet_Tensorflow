# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys

import PIL.Image as Image

sys.path.append('../../')
import xml.etree.cElementTree as ET
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


def read_od_json_gtbox_and_label(od_path):
    with open(od_path) as f:
        data = json.load(f)
    img_width = data['height']
    img_height = data['width']
    box_list = []
    bounding_boxes = data['boundingBoxes']
    for bbox in bounding_boxes:
        tmp_box = []
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
    data_dir = '/home/faisal/python-microservices/image-recognition/image_recognition/tmp/penetrations_05192020_val_converted'
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
    for count, od in enumerate(glob.glob(od_path + '/*.ground_truth.od.json')):

        img_name = od.split('/')[-1].split('.')[0] + img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_od_json_gtbox_and_label(od)
        # if img_height != 600 or img_width != 600:
        #     continue

        # img = np.array(Image.open(img_path))
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        image = np.asarray(image)
        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(int(image.shape[0])),
            'img_width': _int64_feature(int(image.shape[1])),
            'img': _bytes_feature(encoded_jpg),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(od_path + '/*.ground_truth.od.json')))
        if count==5:
            break
    print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pv_data_to_tfrecord()
