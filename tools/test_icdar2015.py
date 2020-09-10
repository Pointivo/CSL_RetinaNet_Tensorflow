# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils import tools
from libs.label_name_dict.label_dict import *
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from libs.box_utils import nms_rotate
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms


def worker(gpu_id, images, det_net, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH,
                                                     is_resize=not args.multi_scale)
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
            print('restore model %d ...' % gpu_id)
        for a_img in images:
            raw_img = cv2.imread(a_img)
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            img_short_side_len_list = cfgs.IMG_SHORT_SIDE_LEN if args.multi_scale else [cfgs.IMG_SHORT_SIDE_LEN]

            det_boxes_r_all, det_scores_r_all, det_category_r_all = [], [], []
            for short_size in img_short_side_len_list:
                max_len = cfgs.IMG_MAX_LENGTH
                if raw_h < raw_w:
                    new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), max_len)
                else:
                    new_h, new_w = min(int(short_size * float(raw_h) / raw_w), max_len), short_size
                img_resize = cv2.resize(raw_img, (new_w, new_h))

                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run(
                        [img_batch, detection_boxes_angle, detection_scores, detection_category],
                        feed_dict={img_plac: img_resize[:, :, ::-1]}
                    )

                detected_indices = detected_scores >= cfgs.VIS_SCORE
                detected_scores = detected_scores[detected_indices]
                detected_boxes = detected_boxes[detected_indices]
                detected_categories = detected_categories[detected_indices]

                if detected_boxes.shape[0] == 0:
                    continue
                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                detected_boxes = forward_convert(detected_boxes, False)
                detected_boxes[:, 0::2] *= (raw_w / resized_w)
                detected_boxes[:, 1::2] *= (raw_h / resized_h)
                # detected_boxes = backward_convert(detected_boxes, False)

                det_boxes_r_all.extend(detected_boxes)
                det_scores_r_all.extend(detected_scores)
                det_category_r_all.extend(detected_categories)
            det_boxes_r_all = np.array(det_boxes_r_all)
            det_scores_r_all = np.array(det_scores_r_all)
            det_category_r_all = np.array(det_category_r_all)

            if det_scores_r_all.shape[0] == 0:
                continue

            box_res_rotate_ = []
            label_res_rotate_ = []
            score_res_rotate_ = []

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(det_category_r_all == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_r = det_boxes_r_all[index]
                tmp_label_r = det_category_r_all[index]
                tmp_score_r = det_scores_r_all[index]

                tmp_boxes_r_ = backward_convert(tmp_boxes_r, False)

                try:
                    inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r_),
                                                    scores=np.array(tmp_score_r),
                                                    iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                                    max_output_size=5000)
                except:
                    tmp_boxes_r_ = np.array(tmp_boxes_r_)
                    tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                    tmp[:, 0:-1] = tmp_boxes_r_
                    tmp[:, -1] = np.array(tmp_score_r)
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(cfgs.NMS_IOU_THRESHOLD), 0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            box_res_rotate_ = np.array(box_res_rotate_)
            score_res_rotate_ = np.array(score_res_rotate_)
            label_res_rotate_ = np.array(label_res_rotate_)

            result_dict = {'scales': [1, 1], 'boxes': box_res_rotate_,
                           'scores': score_res_rotate_, 'labels': label_res_rotate_,
                           'image_id': a_img}
            result_queue.put_nowait(result_dict)


def test_icdar2015(det_net, real_test_img_list, gpu_ids, show_box):

    save_path = os.path.join('./test_icdar2015', cfgs.VERSION)
    tools.mkdir(save_path)

    nr_records = len(real_test_img_list)
    pbar = tqdm(total=nr_records)
    gpu_num = len(gpu_ids.strip().split(','))

    nr_image = math.ceil(nr_records / gpu_num)
    result_queue = Queue(500)
    procs = []

    for i, gpu_id in enumerate(gpu_ids.strip().split(',')):
        start = i * nr_image
        end = min(start + nr_image, nr_records)
        split_records = real_test_img_list[start:end]
        proc = Process(target=worker, args=(int(gpu_id), split_records, det_net, result_queue))
        print('process:%d, start:%d, end:%d' % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        res = result_queue.get()
        if res['boxes'].shape[0] == 0:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4 = res['boxes'][:, 0], res['boxes'][:, 1], res['boxes'][:, 2], res['boxes'][:, 3],\
                                         res['boxes'][:, 4], res['boxes'][:, 5], res['boxes'][:, 6], res['boxes'][:, 7]

        x1, y1 = x1 * res['scales'][0], y1 * res['scales'][1]
        x2, y2 = x2 * res['scales'][0], y2 * res['scales'][1]
        x3, y3 = x3 * res['scales'][0], y3 * res['scales'][1]
        x4, y4 = x4 * res['scales'][0], y4 * res['scales'][1]

        boxes = np.transpose(np.stack([x1, y1, x2, y2, x3, y3, x4, y4]))

        if show_box:
            boxes = backward_convert(boxes, False)
            nake_name = res['image_id'].split('/')[-1]
            tools.mkdir(os.path.join(save_path, 'icdar2015_img_vis'))
            draw_path = os.path.join(save_path, 'icdar2015_img_vis', nake_name)
            draw_img = np.array(cv2.imread(res['image_id']), np.float32)

            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                boxes=boxes,
                                                                                labels=res['labels'],
                                                                                scores=res['scores'],
                                                                                method=1,
                                                                                head=np.ones_like(res['scores']) * -1,
                                                                                in_graph=False)
            cv2.imwrite(draw_path, final_detections)

        else:
            tools.mkdir(os.path.join(save_path, 'icdar2015_res'))
            fw_txt_dt = open(os.path.join(save_path, 'icdar2015_res', 'res_{}.txt'.format(res['image_id'].split('/')[-1].split('.')[0])), 'w')

            for box in boxes:
                line = '%d,%d,%d,%d,%d,%d,%d,%d\n' % (box[0], box[1], box[2], box[3],
                                                      box[4], box[5], box[6], box[7])
                fw_txt_dt.write(line)
            fw_txt_dt.close()

        pbar.set_description("Test image %s" % res['image_id'].split('/')[-1])

        pbar.update(1)

    for p in procs:
        p.join()


def eval(num_imgs, test_dir, gpu_ids, show_box):

    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                         if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    if num_imgs == np.inf:
        real_test_img_list = test_imgname_list
    else:
        real_test_img_list = test_imgname_list[: num_imgs]

    csl = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                               is_training=False)
    test_icdar2015(det_net=csl, real_test_img_list=real_test_img_list, gpu_ids=gpu_ids, show_box=show_box)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 strand')

    parser.add_argument('--test_dir', dest='test_dir',
                        help='evaluate imgs dir ',
                        default='/data/yangxue/dataset/ICDAR2015/ch4_test_images', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--eval_num', dest='eval_num',
                        help='the num of eval imgs',
                        default=np.inf, type=int)
    parser.add_argument('--show_box', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")
    eval(args.eval_num,
         test_dir=args.test_dir,
         gpu_ids=args.gpus,
         show_box=args.show_box)



