from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append("../")
from data.io.convert_pv_data_to_tfrecord import read_od_json_gtbox_and_label

from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from libs.label_name_dict.label_dict import *
from libs.networks import build_whole_network
from libs.box_utils import iou_rotate

from libs.box_utils import nms_rotate
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms

NAME_LABEL_MAP = {
    'back_ground': 0,
    'penetration': 1
}


def compute_ap(recall, precision, use_07_metric=False):
    """ ap = compute_ap(recall, precision, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_metrics(detections, annotations, num_bboxes, cls_name, ovthresh = 0.5, use_07_metric = False):
    """
    Top level function that does the PASCAL VOC evaluation.
    :param detections List of lists [img_path, score, x, y, w, h, theta]
    :param annotations
    :param num_bboxes number of total bboxes in gt
    :param cls_name
    :param ovthresh
    :param use_07_metric
    :return: recall, precision and average precision
    """

    class_bboxes = annotations
    image_ids = [detection[0] for detection in detections if detection[-1] == cls_name]  # image paths are image ids
    confidence = np.array([float(detection[1]) for detection in detections if detection[-1] == cls_name])  # scores are confidence values
    BB = np.array([[float(x) for x in detection[2:7]] for detection in detections if detection[-1] == cls_name])  # [x, y, w, h, theta]

    # compute tp and fp
    nd = len(image_ids)  # num of detections.
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]  # reorder the img_name

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_bboxes[image_ids[d]]  # img_id is img_name
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                overlaps = []
                for i in range(len(BBGT)):
                    overlap = iou_rotate.iou_rotate_calculate1(np.array([bb]),
                                                               np.array([BBGT[i]]),
                                                               use_gpu=False)[0]
                    overlaps.append(overlap)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

    # get recall, precison and AP
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(num_bboxes)

    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(recall=recall, precision=precision, use_07_metric=use_07_metric)

    return recall, precision, ap


def detect(det_net, val_images_list, args):
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category, detection_boxes_angle = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch_h=None,
        gtboxes_batch_r=None,
        gt_smooth_label=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model ...')

        all_boxes = []
        pbar = tqdm(val_images_list)
        for img_path in pbar:
            img = cv2.imread(img_path)

            box_res_rotate = []
            label_res_rotate = []
            score_res_rotate = []

            imgH = img.shape[0]
            imgW = img.shape[1]

            img_short_side_len_list = [cfgs.IMG_SHORT_SIDE_LEN]

            if imgH < args.h_len:
                temp = np.zeros([args.h_len, imgW, 3], np.float32)
                temp[0:imgH, :, :] = img
                img = temp
                imgH = args.h_len

            if imgW < args.w_len:
                temp = np.zeros([imgH, args.w_len, 3], np.float32)
                temp[:, 0:imgW, :] = img
                img = temp
                imgW = args.w_len

            for hh in range(0, imgH, args.h_len - args.h_overlap):
                if imgH - hh - 1 < args.h_len:
                    hh_ = imgH - args.h_len
                else:
                    hh_ = hh

                for ww in range(0, imgW, args.w_len - args.w_overlap):
                    if imgW - ww - 1 < args.w_len:
                        ww_ = imgW - args.w_len
                    else:
                        ww_ = ww
                    src_img = img[hh_:(hh_ + args.h_len), ww_:(ww_ + args.w_len), :]

                    for short_size in img_short_side_len_list:
                        max_len = cfgs.IMG_MAX_LENGTH
                        if args.h_len < args.w_len:
                            new_h, new_w = short_size, min(int(short_size * float(args.w_len) / args.h_len), max_len)
                        else:
                            new_h, new_w = min(int(short_size * float(args.h_len) / args.w_len), max_len), short_size
                        img_resize = cv2.resize(src_img, (new_w, new_h))

                        resized_img, det_boxes_r_, det_scores_r_, det_category_r_ = \
                            sess.run(
                                [img_batch, detection_boxes_angle, detection_scores, detection_category],
                                feed_dict={img_plac: img_resize[:, :, ::-1]}
                            )

                        resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                        src_h, src_w = src_img.shape[0], src_img.shape[1]

                        if len(det_boxes_r_) > 0:
                            det_boxes_r_ = forward_convert(det_boxes_r_, False)
                            det_boxes_r_[:, 0::2] *= (src_w / resized_w)
                            det_boxes_r_[:, 1::2] *= (src_h / resized_h)
                            det_boxes_r_ = backward_convert(det_boxes_r_, False)

                            for ii in range(len(det_boxes_r_)):
                                box_rotate = det_boxes_r_[ii]
                                box_rotate[0] = box_rotate[0] + ww_
                                box_rotate[1] = box_rotate[1] + hh_
                                box_res_rotate.append(box_rotate)
                                label_res_rotate.append(det_category_r_[ii])
                                score_res_rotate.append(det_scores_r_[ii])

            box_res_rotate = np.array(box_res_rotate)
            label_res_rotate = np.array(label_res_rotate)
            score_res_rotate = np.array(score_res_rotate)

            box_res_rotate_ = []
            label_res_rotate_ = []
            score_res_rotate_ = []
            threshold = {'penetration': 0.01}

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res_rotate == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_r = box_res_rotate[index]
                tmp_label_r = label_res_rotate[index]
                tmp_score_r = score_res_rotate[index]

                tmp_boxes_r = np.array(tmp_boxes_r)
                tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_r
                tmp[:, -1] = np.array(tmp_score_r)

                try:
                    inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
                                                    scores=np.array(tmp_score_r),
                                                    iou_threshold=threshold[LABEL_NAME_MAP[sub_class]],
                                                    max_output_size=500)
                except:
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(threshold[LABEL_NAME_MAP[sub_class]]), 0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            result_dict = {'boxes': np.array(box_res_rotate_), 'scores': np.array(score_res_rotate_),
                           'labels': np.array(label_res_rotate_), 'image_id': img_path}
            all_boxes.append(result_dict)
            pbar.set_description("Eval image %s" % img_path)
        return all_boxes


def run_validation(test_dir, step, args):
    val_images_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                       if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(val_images_list) != 0, 'val_dir has 0 images.' \
                                      ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    csl_val = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)
    all_boxes = detect(det_net=csl_val, val_images_list=val_images_list[:5], args=args)

    detections = []
    for boxes in all_boxes:
        for i in range(len(boxes['boxes'])):
            detection_image = []
            detection_image.append(boxes['image_id'])
            detection_image.append(boxes['scores'][i])
            detection_image.append(boxes['boxes'][i][0])
            detection_image.append(boxes['boxes'][i][1])
            detection_image.append(boxes['boxes'][i][2])
            detection_image.append(boxes['boxes'][i][3])
            detection_image.append(boxes['boxes'][i][4])
            detection_image.append(LABEL_NAME_MAP[boxes['labels'][i]])
            detections.append(detection_image)

    class_bboxes = {}
    num_bboxes = 0
    image_paths = []
    cls_name = 'penetration'
    for detection in detections:
        if detection[0] not in image_paths:
            image_paths.append(detection[0])

    for image_path in image_paths:
        img_name = image_path.split('/')[-1][:-4]
        ann_path = test_dir + '/' + img_name + '.ground_truth.od.json'
        _, _, gtbboxes = read_od_json_gtbox_and_label(ann_path)
        gtbboxes = backward_convert(gtbboxes, with_label=True)  # [x, y, w, h, theta, (label)]
        R = [gtbox for gtbox in gtbboxes if LABEL_NAME_MAP[gtbox[-1]] == cls_name]
        num_bboxes = num_bboxes + len(R)
        bbox = np.array([x[:-1] for x in R])  # [x, y, w, h, theta]
        det = [False] * len(R)  # det means that gtboxes has already been detected
        class_bboxes[image_path] = {'bbox': bbox, 'det': det}

    recall, precision, ap = compute_metrics(detections=detections, annotations=class_bboxes, num_bboxes=num_bboxes,
                                            cls_name='penetration', ovthresh=0.2)

    # writing these metrics to tensorboard
    summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
    summary_writer = tf.summary.FileWriter(summary_path)
    summary = tf.Summary(value=[tf.Summary.Value(tag='validation_mAP', simple_value=ap)])
    summary_writer.add_summary(summary=summary, global_step=step)
    summary_writer.flush()


def get_latest_step(file_path):
    if os.path.isfile(file_path):
        checkpoint = open(file_path, 'r')
        lines = checkpoint.readlines()
        first_line = lines[0]
        output = re.search('PENETRATION_(.+?)model.ckpt', first_line)
        step = output.group(1)
        return step


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='validation image...you need provide the val dir')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='val data path', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu id ',
                        default='0', type=str)
    parser.add_argument('--h_len', dest='h_len',
                        help='image height',
                        default=600, type=int)
    parser.add_argument('--w_len', dest='w_len',
                        help='image width',
                        default=600, type=int)
    parser.add_argument('--h_overlap', dest='h_overlap',
                        help='height overlap',
                        default=150, type=int)
    parser.add_argument('--w_overlap', dest='w_overlap',
                        help='width overlap',
                        default=150, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    checkpoint_file = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION, 'checkpoint')
    step = get_latest_step(file_path=checkpoint_file)
    run_validation(test_dir=args.data_dir, step=step, args=args)
