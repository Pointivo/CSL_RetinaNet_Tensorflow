# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import tensorflow as tf

sys.path.append("../")

from libs.networks.build_whole_network import DetectionNetwork
from help_utils import tools
from libs.label_name_dict.label_dict import *
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from libs.box_utils import nms_rotate
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms
from image_recognition.app.dvo.bbox_2d.oriented_bbox_2d import OrientedBbox2D
from image_recognition.app.dvo.ground_truths.object_detection import ObjectDetectionLabeledData
from image_recognition.app.data.dataset import Dataset, DataSignatures


def _scale_bbox(bbox: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
    x1, y1, x2, y2, x3, y3, x4, y4, class_label = bbox
    x1, x2, x3, x4 = x1 * x_scale, x2 * x_scale, x3 * x_scale, x4 * x_scale
    y1, y2, y3, y4 = y1 * y_scale, y2 * y_scale, y3 * y_scale, y4 * y_scale
    return np.array([x1, y1, x2, y2, x3, y3, x4, y4, class_label], dtype=np.int32)


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


def get_checkpoint_path_from_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_paths = tf.train.get_checkpoint_state(str(checkpoint_dir)).all_model_checkpoint_paths
    checkpoint_paths = [Path(checkpoint_path) for checkpoint_path in checkpoint_paths]
    # paths in checkpoint files might be different from checkpoint_dir
    checkpoint_paths = [checkpoint_dir / path.name for path in checkpoint_paths]
    assert len(checkpoint_paths) == 1, f'There should be only 1 checkpoint in the checkpoint directory'
    return checkpoint_paths[0]


def get_csl_prediction_results(gpu_id: int, images: List[str], det_net: DetectionNetwork, rotated_iou_thresh: float,
                               checkpoint_path: Optional[str] = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    det_boxes, det_scores, det_category, det_box_angles = det_net.build_whole_detection_network(
        input_img_batch=img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gt_smooth_label=None)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    restorer, restore_ckpt = det_net.get_restorer(checkpoint_path=checkpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    prediction_results = []
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        assert restorer is not None, f'Restorer is None. Something went wrong!!!'
        restorer.restore(sess, restore_ckpt)
        print(f'Restored model from {restore_ckpt}.')

        for i, img_path in enumerate(images):
            print(f'[{i}/{len(images)}]\t Running predictions for image {Path(img_path).name}.')
            img = cv2.imread(img_path)
            orig_h, orig_w, _ = img.shape
            img = _pad_image_with_zeros(image_np=img)
            padded_h, padded_w, _ = img.shape

            img = cv2.resize(img, (cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN), interpolation=cv2.INTER_AREA)
            resized_h, resized_w, _ = img.shape

            resized_img, box_res_rotate, score_res_rotate, label_res_rotate = sess.run(
                [img_batch, det_box_angles, det_scores, det_category], feed_dict={img_plac: img[:, :, ::-1]})
            assert len(resized_img) == 1, f'Something went wrong! There should be a single image in batch. Got ' \
                                          f'{resized_img.shape}.'
            assert resized_img[0].shape == img.shape, f'Something went wrong! The shape of image being fed should be ' \
                                                      f'equal to the image returned.'
            if box_res_rotate.size > 0:
                box_res_rotate = forward_convert(box_res_rotate, False)
                box_res_rotate[:, 0::2] *= (padded_w / resized_w)
                box_res_rotate[:, 1::2] *= (padded_h / resized_h)
                box_res_rotate = backward_convert(box_res_rotate, False)

            box_res_rotate_ = []
            label_res_rotate_ = []
            score_res_rotate_ = []

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
                                                    iou_threshold=rotated_iou_thresh,
                                                    max_output_size=500)
                except:
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(rotated_iou_thresh), 0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            result_dict = {'boxes': np.array(box_res_rotate_), 'scores': np.array(score_res_rotate_),
                           'labels': np.array(label_res_rotate_), 'image_id': img_path}
            prediction_results.append(result_dict)
    return prediction_results


def get_image_paths_from_dataset_dir(dataset_dir: str, eval_num: int) -> List[str]:
    dataset_image_paths = [os.path.join(dataset_dir, img_name) for img_name in os.listdir(dataset_dir)
                           if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(dataset_image_paths) > 0, 'dataset_dir has no imgs: we only support .jpg, .png, and .tiff image formats.'

    dataset_image_paths = dataset_image_paths if eval_num == np.inf else dataset_image_paths[:args.eval_num]
    return dataset_image_paths


def get_class_names_from_class_labels(class_labels: List[int], class_name_to_label_map: Dict[str, int]) -> List[str]:
    class_names = []
    for class_label in class_labels:
        for cls_name, cls_lbl in class_name_to_label_map.items():
            if cls_lbl == class_label:
                class_names.append(cls_name)
    assert len(class_labels) == len(class_names)
    return class_names


def save_detections_for_images(det_net: DetectionNetwork, class_name_to_label_map: Dict[str, int],
                               args: argparse.Namespace):
    image_paths = get_image_paths_from_dataset_dir(dataset_dir=args.dataset_dir, eval_num=args.eval_num)
    checkpoint_path = get_checkpoint_path_from_checkpoint_dir(checkpoint_dir=Path(args.checkpoint_dir))
    prediction_results = get_csl_prediction_results(gpu_id=args.gpu, images=image_paths, det_net=det_net,
                                                    rotated_iou_thresh=args.rotated_iou_thresh,
                                                    checkpoint_path=str(checkpoint_path))

    if args.mode == 'vis':
        print(f'Saving visualizations in {args.save_vis_dir}.')
        assert args.save_vis_dir is not None, f'save_vis_dir cannot be None if mode is set to vis'
        for prediction_result in prediction_results:
            image_name = Path(prediction_result['image_id']).name
            tools.mkdir(args.save_vis_dir)
            draw_path = os.path.join(args.save_vis_dir, image_name)

            detected_indices = prediction_result['scores'] >= args.conf_thresh
            detected_scores = prediction_result['scores'][detected_indices]
            detected_boxes = prediction_result['boxes'][detected_indices]
            detected_categories = prediction_result['labels'][detected_indices]
            img = cv2.imread(prediction_result['image_id']) / 255
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(img,
                                                                                boxes=detected_boxes,
                                                                                labels=detected_categories,
                                                                                scores=detected_scores,
                                                                                method=1,
                                                                                head=np.ones_like(detected_scores) * -1,
                                                                                in_graph=False)
            cv2.imwrite(draw_path, final_detections)
    elif args.mode == 'save_pred_od':
        print(f"Saving .prediction.od.json's in dataset_dir.")
        dataset = Dataset(dataset_dir=Path(args.dataset_dir))
        for prediction_result in prediction_results:
            detected_indices = prediction_result['scores'] >= args.conf_thresh
            confidence_scores = prediction_result['scores'][detected_indices]
            detected_boxes = prediction_result['boxes'][detected_indices]
            detected_categories = prediction_result['labels'][detected_indices]

            class_names = get_class_names_from_class_labels(class_labels=detected_categories,
                                                            class_name_to_label_map=class_name_to_label_map)

            image_name = Path(prediction_result['image_id']).stem
            rotated_boxes = forward_convert(detected_boxes, with_label=False)

            ic = dataset.get_data_from_file(data_signature=DataSignatures.ic, file_name_stem=Path(image_name).stem)
            obboxes2d = []
            for rotated_box, score, class_name in zip(rotated_boxes, confidence_scores, class_names):
                x1, y1, x2, y2, x3, y3, x4, y4 = rotated_box
                obbox2d = OrientedBbox2D(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, confidence_score=score,
                                         class_label=class_name)
                obboxes2d.append(obbox2d)
            pred_od = ObjectDetectionLabeledData(image_name=image_name, bounding_boxes=obboxes2d, width=ic.width,
                                                 height=ic.height)
            dataset.save_od_json(od=pred_od, data_signature=DataSignatures.od_predicted)
    else:
        raise AssertionError(f"mode is not supported: Got mode={args.mode}. Set to either 'vis' or 'save_pred_od'.")


def detect(args: argparse.Namespace, class_name_to_label_map: Dict[str, int]):
    csl_net = DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=False)
    save_detections_for_images(det_net=csl_net, class_name_to_label_map=class_name_to_label_map, args=args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', dest='dataset_dir', help='dataset with images', required=True, type=str)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', required=True, type=str)
    parser.add_argument('--mode', dest='mode', help='set to either vis or save_pred_od', required=True, type=str)
    parser.add_argument('--conf_thresh', dest='conf_thresh', default=0.1, type=float)
    parser.add_argument('--rotated_iou_thresh', dest='rotated_iou_thresh', default=0.1, type=float)
    parser.add_argument('--save_vis_dir', dest='save_vis_dir', help='img visualize dir', default=None, type=str)
    parser.add_argument('--eval_num', dest='eval_num', help='the num of eval imgs', default=np.inf, type=int)
    parser.add_argument('--gpu', dest='gpu', help='gpu id', default='0', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(f'Called with args = {args}')
    class_name_to_label_map = {'back_ground': 0, 'penetration': 1}
    detect(args=args, class_name_to_label_map=class_name_to_label_map)
