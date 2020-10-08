from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append("../")

from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from libs.label_name_dict.label_dict import *
from libs.networks.build_whole_network import DetectionNetwork
from libs.box_utils import iou_rotate

from libs.box_utils import nms_rotate
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms


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


def compute_metrics(detections, annotations, num_bboxes, cls_name, ovthresh=0.5, use_07_metric=False):
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
    confidence = np.array(
        [float(detection[1]) for detection in detections if detection[-1] == cls_name])  # scores are confidence values
    BB = np.array([[float(x) for x in detection[2:7]] for detection in detections
                   if detection[-1] == cls_name])  # [x, y, w, h, theta]

    # compute tp and fp
    nd = len(image_ids)  # num of detections.
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        confidence = confidence[sorted_ind]
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
                                                               use_gpu=False)[0][0]
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

    return recall, precision, ap, confidence


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


def get_csl_prediction_results(images: List[str], det_net: DetectionNetwork, checkpoint_path: Path,
                               args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

    restorer, checkpoint_path = det_net.get_restorer(checkpoint_path=checkpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    prediction_results = []
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        assert restorer is not None, f'Restorer is None. Something went wrong!!!'
        restorer.restore(sess, str(checkpoint_path))

        for img_path in tqdm(images, desc='Running predictions.'):
            # reading image
            img = cv2.imread(img_path)
            # padding image with zeros to make it square
            img = _pad_image_with_zeros(image_np=img)
            padded_h, padded_w, _ = img.shape
            # resizing to IMG_SHORT_SIDE_LEN x IMG_SHORT_SIDE_LEN
            img = cv2.resize(img, (cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_SHORT_SIDE_LEN), interpolation=cv2.INTER_AREA)
            resized_h, resized_w, _ = img.shape

            # running prediction
            resized_img, box_res_rotate, score_res_rotate, label_res_rotate = sess.run(
                [img_batch, det_box_angles, det_scores, det_category], feed_dict={img_plac: img[:, :, ::-1]})
            assert len(resized_img) == 1, f'Something went wrong! There should be a single image in batch. Got ' \
                                          f'{resized_img.shape}.'
            assert resized_img[0].shape == img.shape, f'Something went wrong! The shape of image being fed should be ' \
                                                      f'equal to the image returned.'

            # rescaling predictions to original image height and width
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
                                                    iou_threshold=args.rotated_iou_thresh,
                                                    max_output_size=500)
                except:
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(args.rotated_iou_thresh), 0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            score_res_rotate_ = np.array(score_res_rotate_)
            box_res_rotate_ = np.array(box_res_rotate_)
            label_res_rotate_ = np.array(label_res_rotate_)
            result_dict = {'boxes': box_res_rotate_, 'scores': score_res_rotate_,
                           'labels': label_res_rotate_, 'image_id': img_path}
            prediction_results.append(result_dict)
    return prediction_results


def _get_gt_class_bboxes_from_pv_dataset(dataset_dir: Path, class_name: str, class_name_to_label_map: Dict[str, int]):
    class_bboxes = {}
    num_bboxes = 0
    image_paths = [str(p) for p in dataset_dir.glob('*') if p.name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    for image_path in image_paths:
        od_file_path = dataset_dir / f'{Path(image_path).stem}.ground_truth.od.json'
        gtbboxes = _get_bounding_boxes_from_od_json_file(od_file_path=od_file_path,
                                                         class_name_to_label_map=class_name_to_label_map)
        gtbboxes = backward_convert(gtbboxes, with_label=True)  # [x, y, w, h, theta, (label)]
        R = [gtbox for gtbox in gtbboxes if LABEL_NAME_MAP[gtbox[-1]] == class_name]
        num_bboxes = num_bboxes + len(R)
        bbox = np.array([x[:-1] for x in R])  # [x, y, w, h, theta]
        det = [False] * len(R)  # det means that gtboxes has already been detected
        class_bboxes[image_path] = {'bbox': bbox, 'det': det}
    return class_bboxes, num_bboxes


def _get_detections_from_csl_predictions(csl_predictions: List) -> List:
    detections = []
    for boxes in csl_predictions:
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
    return detections


def _write_tensorboard_summaries(step: int, ap50: float, optimal_conf_thresh: float, optimal_precision: float,
                                 optimal_recall: float, optimal_f1_score: float):
    summary_file_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
    summary_writer = tf.summary.FileWriter(summary_file_path)

    summary = tf.Summary(value=[tf.Summary.Value(tag='AP50', simple_value=ap50)])
    summary_writer.add_summary(summary=summary, global_step=step)

    summary = tf.Summary(value=[tf.Summary.Value(tag='optimal_conf_thresh', simple_value=optimal_conf_thresh)])
    summary_writer.add_summary(summary=summary, global_step=step)

    summary = tf.Summary(value=[tf.Summary.Value(tag='optimal_precision', simple_value=optimal_precision)])
    summary_writer.add_summary(summary=summary, global_step=step)

    summary = tf.Summary(value=[tf.Summary.Value(tag='optimal_recall', simple_value=optimal_recall)])
    summary_writer.add_summary(summary=summary, global_step=step)

    summary = tf.Summary(value=[tf.Summary.Value(tag='optimal_f1_score', simple_value=optimal_f1_score)])
    summary_writer.add_summary(summary=summary, global_step=step)

    summary_writer.flush()


def _print_metrics(ap50: float, optimal_conf_thresh: float, optimal_precision: float, optimal_recall: float,
                   optimal_f1_score: float):
    print(f'AP50\t\t\t\t = {ap50:0.3f}')
    print(f'optimal_conf_thresh\t = {optimal_conf_thresh:0.3f}')
    print(f'optimal_f1_score\t = {optimal_f1_score:0.3f}')
    print(f'optimal_precision\t = {optimal_precision:0.3f}')
    print(f'optimal_recall\t\t = {optimal_recall:0.3f}')


def run_validation(dataset_dir: Path, class_name_to_label_map: Dict[str, int], checkpoint_path: Path,
                   args: argparse.Namespace):
    image_paths = [str(p) for p in dataset_dir.glob('*') if p.name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(image_paths) != 0, f'No images found in {dataset_dir}.'
    step = get_step_from_checkpoint_path(checkpoint_path=checkpoint_path)
    print(f'Running validation for checkpoint {checkpoint_path.name} (step_number={step}).')

    csl_net = DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=False)
    csl_predictions = get_csl_prediction_results(det_net=csl_net, images=image_paths, checkpoint_path=checkpoint_path,
                                                 args=args)
    detections = _get_detections_from_csl_predictions(csl_predictions=csl_predictions)

    class_names = [class_name for class_name in list(class_name_to_label_map) if class_name != 'back_ground']
    assert len(class_names) == 1, f'Only single class metrics are supported currently. Got classes={class_names}'
    class_name = class_names[0]
    class_bboxes, num_bboxes = _get_gt_class_bboxes_from_pv_dataset(dataset_dir, class_name=class_name,
                                                                    class_name_to_label_map=class_name_to_label_map)

    recall, precision, ap, confidences = compute_metrics(detections=detections, annotations=class_bboxes,
                                                         num_bboxes=num_bboxes, cls_name=class_name,
                                                         ovthresh=args.tp_iou_thresh)
    assert recall.shape == precision.shape, f'Something went wrong!!! Recall and precisions donot have the same shape.'
    f1_score = 2 * recall * precision / (recall + precision + np.finfo(np.float64).eps)
    if f1_score.size > 0:
        max_f1_score_idx = np.argmax(f1_score)
        optimal_conf_thresh = confidences[max_f1_score_idx]
        optimal_f1_score = f1_score[max_f1_score_idx]
        optimal_precision = precision[max_f1_score_idx]
        optimal_recall = recall[max_f1_score_idx]
    else:
        optimal_f1_score = optimal_conf_thresh = optimal_precision = optimal_recall = 0

    # writing these metrics to tensorboard
    _write_tensorboard_summaries(step=step, ap50=ap, optimal_f1_score=optimal_f1_score, optimal_recall=optimal_recall,
                                 optimal_conf_thresh=optimal_conf_thresh, optimal_precision=optimal_precision)
    _print_metrics(ap50=ap, optimal_f1_score=optimal_f1_score, optimal_recall=optimal_recall,
                   optimal_conf_thresh=optimal_conf_thresh, optimal_precision=optimal_precision)

def get_all_checkpoints_from_checkpoint_dir(checkpoint_dir: Path) -> List[Path]:
    checkpoint_paths = tf.train.get_checkpoint_state(str(checkpoint_dir)).all_model_checkpoint_paths
    assert checkpoint_paths is not None, 'Something went wrong!!! Make sure checkpoint file is correct and present.'
    return [Path(checkpoint_path) for checkpoint_path in checkpoint_paths]


def get_step_from_checkpoint_path(checkpoint_path: Path) -> int:
    checkpoint_stem_name = checkpoint_path.stem
    assert cfgs.DATASET_NAME in checkpoint_stem_name, f"checkpoint {checkpoint_path.name} does not belong to " \
                                                      f"Dataset '{cfgs.DATASET_NAME}'."
    checkpoint_suffix = checkpoint_stem_name.split(sep=f'{cfgs.DATASET_NAME}_')[1]
    assert 'model' in checkpoint_suffix, f'checkpoint name should be {cfgs.DATASET_NAME}_xxmodel.ckpt, but got ' \
                                         f'{checkpoint_path.name}.'
    return int(checkpoint_suffix.split('model')[0])


def run_validation_on_dataset(args: argparse.Namespace, class_name_to_label_map: Dict[str, int]):
    dataset_dir = Path(args.dataset_dir)
    checkpoint_dir = Path(cfgs.TRAINED_CKPT) / cfgs.VERSION
    all_checkpoint_paths = get_all_checkpoints_from_checkpoint_dir(checkpoint_dir=checkpoint_dir)
    print(f'Found {len(all_checkpoint_paths)} number of checkpoints under {checkpoint_dir}.')

    for checkpoint_path in all_checkpoint_paths:
        print('+-' * 100)
        run_validation(dataset_dir=dataset_dir, checkpoint_path=checkpoint_path, args=args,
                       class_name_to_label_map=class_name_to_label_map)
        tf.reset_default_graph()
        print('+-' * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calculating validation metrics for model checkpoints.')
    parser.add_argument('--dataset_dir', dest='dataset_dir', help='val data path', type=str)
    parser.add_argument('--gpu', dest='gpu', help='gpu id ', default='0', type=str)
    parser.add_argument('--rotated_iou_thresh', dest='rotated_iou_thresh', default=0.1, type=float)
    parser.add_argument('--tp_iou_thresh', dest='tp_iou_thresh', default=0.5, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:', args)

    class_name_to_label_map = {'back_ground': 0, 'penetration': 1}
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_validation_on_dataset(args=args, class_name_to_label_map=class_name_to_label_map)
