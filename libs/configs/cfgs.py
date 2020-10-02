# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import math
import os

import tensorflow as tf

"""
v27 + data aug. + MS + atan
This is your result for task 1:

    mAP: 0.7028884684751898
    ap of each class: plane:0.8667564161146087,
    baseball-diamond:0.8253459467920615,
    bridge:0.4897846778195108,
    ground-track-field:0.747228457008536,
    small-vehicle:0.6405610641223431,
    large-vehicle:0.49705671204943547,
    ship:0.6335849234698555,
    tennis-court:0.893144800266831,
    basketball-court:0.8626350724764835,
    storage-tank:0.8609387992114981,
    soccer-ball-field:0.6694961012635908,
    roundabout:0.6346595227564863,
    harbor:0.6509623790910654,
    swimming-pool:0.6949999394263442,
    helicopter:0.5761722152591966

The submitted information is :

Description: RetinaNet_DOTA_4x_20200202_162w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


PV Note: This config is modified version of "cfgs_res152_dota_v36.py". The config was modified for oriented 
penetrations detection training.
"""

# ------------------------------------------------
VERSION = 'RetinaNet_Penetration_4x_02_Oct_2020'
NET_NAME = 'resnet152_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 50
SMRY_ITER = 2000
SAVE_WEIGHTS_INTE = 2000  # no. of training images ~ 16,000
SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = 1

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
DECAY_STEP = [int(16000*13), int(16000*17), int(16000*21)]
MAX_ITERATION = int(16000*21)
WARM_SETP = 16000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'PENETRATION'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 1200
IMG_MAX_LENGTH = 1200
CLASS_NUM = 1
LABEL_TYPE = 0
RADUIUS = 6
OMEGA = 1

IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1 / 4.0, 4.0, 1 / 6.0, 6.0]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = False
ANGLE_RANGE = 180  # 180 or 90

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

