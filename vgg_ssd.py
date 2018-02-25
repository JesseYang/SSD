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

class VGGSSD(SSDModel):

    def get_logits(self, image):

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()):
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

            if get_tf_version_number() >= 1.5:
                conv6 = Conv2D('conv6', conv5, 1024, 3, dilation_rate=6)
            else:
                filter_shape = [3, 3, 512, 1024]
                W_init = tf.contrib.layers.xavier_initializer()
                b_init = tf.constant_initializer()
                W = tf.get_variable('W', filter_shape, initializer=W_init)
                b = tf.get_variable('b', [1024], initializer=b_init)
                conv6 = tf.nn.atrous_conv2d(conv5, W, rate=6, padding="SAME")
                conv6 = tf.nn.bias_add(conv6, b)
                conv6 = tf.nn.relu(conv6, name='conv6')
            conv7 = Conv2D('conv7', conv6, 1024, 1)

            # 10
            conv8 = (LinearWrap(conv7)
                    .Conv2D('conv8_1', 256, 1)
                    .Conv2D('conv8_2', 512, 3, stride=2)())

            # 5
            conv9 = (LinearWrap(conv8)
                    .Conv2D('conv9_1', 128, 1)
                    .Conv2D('conv9_2', 256, 3, stride=2)())

            # 3
            conv10 = (LinearWrap(conv9)
                     .Conv2D('conv10_1', 128, 1)
                     .Conv2D('conv10_2', 256, 3, padding="VALID")())

            # 1
            conv11 = (LinearWrap(conv10)
                     .Conv2D('conv11_1', 128, 1)
                     .Conv2D('conv11_2', 256, 3, padding="VALID")())

        conv4_3_shape = conv4_3.get_shape()
        if self.data_format == 'NHWC':
            norm_dim = 3
            scale_shape = conv4_3_shape[-1:]
        else:
            norm_dim = 1
            scale_shape = (conv4_3_shape[1])
        scale = variables.model_variable('conv4_3_scale',
                                         shape=scale_shape,
                                         dtype=conv4_3.dtype.base_dtype,
                                         # initializer=init_ops.ones_initializer(),
                                         initializer=tf.constant_initializer(20),
                                         trainable=True)
        if self.data_format == 'NCHW':
            scale = tf.expand_dims(scale, axis=-1)
            scale = tf.expand_dims(scale, axis=-1)


        conv4_3_norm = tf.nn.l2_normalize(conv4_3, norm_dim)
        conv4_3_scale = tf.multiply(conv4_3_norm, scale)

        features = [conv4_3_scale, conv7, conv8, conv9, conv10, conv11]
        return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--itr', help='number of iterations', type=int, default=120000)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = VGGSSD()
    if args.flops:
        input_desc = [
            InputDesc(tf.uint8, [2, cfg.img_h, cfg.img_w, 3], 'input'),
            InputDesc(tf.int32, [2, cfg.tot_anchor_num], 'conf_label'),
            InputDesc(tf.float32, [2, cfg.tot_anchor_num, 4], 'loc_label'),
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
