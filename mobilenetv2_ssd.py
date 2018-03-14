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


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, nl=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return nl(conv, name='output')

def BN(x, name):
    return BatchNorm('bn', x)

def BNReLU6(x, name):
    x = BN(x, 'bn')
    return tf.nn.relu6(x, name=name)


class SSDLite(SSDModel):

    def get_logits(self, image):

        def bottleneck_v2(l, t, out_channel, stride=1):
            in_shape = l.get_shape().as_list()

            in_channel = in_shape[1] if self.data_format == "NCHW" else in_shape[3]
            shortcut = l
            l = Conv2D('conv1', l, t*in_channel, 1, nl=BNReLU6)
            expansion = l
            l = DepthConv('depthconv', l, t*in_channel, 3, stride=stride, nl=BNReLU6)
            l = Conv2D('conv2', l, out_channel, 1, nl=BN)
            if stride == 1 and out_channel == in_channel:
                l = l + shortcut
            return l, expansion

        with argscope([Conv2D, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope([Conv2D], use_bias=False):

            # 320
            l = Conv2D('covn1', image, 32, 3, stride=2, nl=BNReLU)
            # 160
            with tf.variable_scope('bottleneck1'):
                l, _ = bottleneck_v2(l, out_channel=16, t=1, stride=1)

            # 80 after depth-conv of first block
            with tf.variable_scope('bottleneck2'):
                for j in range(2):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=24, t=6, stride=2 if j == 0 else 1)

            # 40 after depth-conv of first block
            with tf.variable_scope('bottleneck3'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=32, t=6, stride=2 if j == 0 else 1)
            
            # 20 after depth-conv of first bolck
            with tf.variable_scope('bottleneck4'):
                for j in range(4):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=64, t=6, stride=2 if j == 0 else 1)
            
            # 20
            with tf.variable_scope('bottleneck5'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=96, t=6, stride=1)
            
            # 10 after depth-conv of first bolck
            # the next layer is the 15th layer
            with tf.variable_scope('bottleneck6'):
                with tf.variable_scope('block{}'.format(0)):
                    l, feat_1 = bottleneck_v2(l, out_channel=160, t=6, stride=2)
                for j in range(2):
                    with tf.variable_scope('block{}'.format(j+1)):
                        l, _ = bottleneck_v2(l, out_channel=160, t=6, stride=1)

            # 10
            with tf.variable_scope('bottleneck7'):
                l, _ = bottleneck_v2(l, out_channel=320, t=6, stride=1)

            if cfg.freeze_backbone == True:
                feat_1 = tf.stop_gradient(feat_1)
                l = tf.stop_gradient(l)

            # feat_2 = Conv2D('conv2', l, 1280, 1, nl=BNReLU)
            feat_2 = Conv2D('extra_conv1_1', l, 256, 1, nl=BNReLU)

            # the extra layers
            # 5
            feat_3 = (LinearWrap(feat_2)
                     # .Conv2D('extra_conv1_1', 256, 1)
                     .DepthConv('extra_conv1_2', 256, 3, stride=2)
                     .Conv2D('extra_conv1_3', 512, 1)())

            # 3
            feat_4 = (LinearWrap(feat_3)
                     .Conv2D('extra_conv2_1', 128, 1)
                     .DepthConv('extra_conv2_2', 128, 3, stride=2)
                     .Conv2D('extra_conv2_3', 256, 1)())

            # 2
            feat_5 = (LinearWrap(feat_4)
                     .Conv2D('extra_conv3_1', 128, 1)
                     .DepthConv('extra_conv3_2', 128, 3, stride=2)
                     .Conv2D('extra_conv3_3', 256, 1)())

            # 1
            feat_6 = (LinearWrap(feat_5)
                     .Conv2D('extra_conv4_1', 128, 1)
                     .DepthConv('extra_conv4_2', 128, 3, stride=2)
                     .Conv2D('extra_conv4_3', 256, 1)())

        features = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6]
        return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--itr', help='number of iterations', type=int, default=240000)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = SSDLite(data_format='NCHW')
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
