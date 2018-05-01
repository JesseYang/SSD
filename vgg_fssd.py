#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pdb
import cv2
import sys
import argparse
import numpy as np
import os
import shutil
import multiprocessing
import json

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

try:
    from .cfgs.config import cfg
    from .evaluate import do_python_eval
    from .ssd_utils import SSDModel, get_data, get_config
except Exception:
    from cfgs.config import cfg
    from evaluate import do_python_eval
    from ssd_utils import SSDModel, get_data, get_config


class VGGFSSD(SSDModel):

    # def __init__(self, data_format="NHWC"):
    #     super(SSDModel, self).__init__()
    #     self.data_format = data_format

    def get_logits(self, image):

        with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()):
            conv4_3 = (LinearWrap(image)
                      .Conv2D('conv1_1', 64)
                      .Conv2D('conv1_2', 64)
                      .MaxPooling('pool1', 2, padding="SAME")
                      # 150
                      .Conv2D('conv2_1', 128)
                      .Conv2D('conv2_2', 128)
                      .MaxPooling('pool2', 2, padding="SAME")
                      # 75
                      .Conv2D('conv3_1', 256)
                      .Conv2D('conv3_2', 256)
                      .Conv2D('conv3_3', 256)
                      .MaxPooling('pool3', 2, padding="SAME")
                      # 38
                      .Conv2D('conv4_1', 512)
                      .Conv2D('conv4_2', 512)
                      .Conv2D('conv4_3', 512)())

            conv5 = (LinearWrap(conv4_3)
                    .MaxPooling('pool4', 2, padding="SAME")
                    # 19
                    .Conv2D('conv5_1', 512)
                    .Conv2D('conv5_2', 512)
                    .Conv2D('conv5_3', 512)
                    .MaxPooling('pool5', 3, 1, padding='SAME')())

            if cfg.freeze_backbone == True:
                conv4_3 = tf.stop_gradient(conv4_3)
                conv5 = tf.stop_gradient(conv5)

            conv6 = Conv2D('conv6', conv5, 1024, 3, dilation_rate=6)
            conv7 = Conv2D('conv7', conv6, 1024, 1)

            # 10
            conv9 = (LinearWrap(conv7)
                    .Conv2D('conv8_1', 256, 1)
                    .Conv2D('conv8_2', 512, 3)
                    .Conv2D('conv9_1', 128, 1)
                    .Conv2D('conv9_2', 256, 3, stride=2)())

            # conv4_3, conv7, and conv8 should be transformed and concatenated to obtain the fused features
            conv4_3_t = Conv2D('conv4_3_t', conv4_3, 256, 1)
            feat_size = tf.shape(conv4_3_t)[2:4] if self.data_format == "NCHW" else tf.shape(conv4_3_t)[1:3]
            conv7_t = Conv2D('conv7_t', conv7, 256, 1)
            conv7_t_u = tf.image.resize_bilinear(conv7_t, feat_size, name='conv7_upsample')
            conv9_t = Conv2D('conv9_t', conv9, 256, 1)
            conv9_t_u = tf.image.resize_bilinear(conv9_t, feat_size, name='conv9_upsample')

            if self.data_format == 'NCHW':
                fuse_features = tf.concat([conv4_3_t, conv7_t_u, conv9_t_u], 1)
            else:
                fuse_features = tf.concat([conv4_3_t, conv7_t_u, conv9_t_u], 3)

            # 38
            fuse_features = BatchNorm('bn_feat', fuse_features)

            # 38
            feature1 = Conv2D('conv_feat1', fuse_features, 512)

            # 19
            feature2 = Conv2D('conv_feat2', feature1, 512, stride=2)

            # 10
            feature3 = Conv2D('conv_feat3', feature2, 256, stride=2)

            # 5
            feature4 = Conv2D('conv_feat4', feature3, 256, stride=2)

            # 3
            feature5 = Conv2D('conv_feat5', feature4, 256, padding='VALID')

            # 1
            feature6 = Conv2D('conv_feat6', feature5, 256, padding='VALID')

        return [feature1, feature2, feature3, feature4, feature5, feature6]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--batch_size_per_gpu', help='batch size per gpu', type=int, default=32)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = VGGFSSD()
    if args.flops:
        input_desc = [
            InputDesc(tf.uint8, [1, cfg.img_h, cfg.img_w, 3], 'input'),
            InputDesc(tf.float32, [1, cfg.max_gt_box_shown, 4], 'gt_bboxes'),
            InputDesc(tf.int32, [1, cfg.tot_anchor_num], 'conf_label'),
            InputDesc(tf.bool, [1, cfg.tot_anchor_num], 'neg_mask'),
            InputDesc(tf.float32, [1, cfg.tot_anchor_num, 4], 'loc_label'),
            InputDesc(tf.float32, [1, 3], 'ori_shape'),
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        # assert args.gpu is not None, "Need to specify a list of gpu for training!"
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()
        config = get_config(args, model)
        if args.gpu != None:
            config.nr_tower = len(args.gpu.split(','))

        if args.load:
            if args.load.endswith('npz'):
                config.session_init = DictRestore(dict(np.load(args.load)))
            else:
                config.session_init = SaverRestore(args.load)

        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
