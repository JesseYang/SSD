# -*- coding: UTF-8 -*-
# File: mobilenetv2.py

import argparse
import numpy as np
import os
import cv2

import tensorflow as tf

import random
from tensorpack import logger, QueueInput, InputDesc, PlaceholderInput, TowerContext
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    get_imagenet_dataflow,
    ImageNetModel, GoogleNetResize, eval_on_ILSVRC12)
from cfgs.config import cfg
TOTAL_BATCH_SIZE = 256


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

class Model(ImageNetModel):
    
    weight_decay = 4e-5
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def get_logits(self, image):

        def bottleneck_v2(l, t, out_channel, stride=1):
            in_shape = l.get_shape().as_list()

            in_channel = in_shape[1] if self.data_format == "NCHW" else in_shape[3]
            shortcut = l
            l = Conv2D('conv1', l, t*in_channel, 1, nl=BNReLU6)
            l = DepthConv('depthconv', l, t*in_channel, 3, stride=stride, nl=BNReLU6)
            l = Conv2D('conv2', l, out_channel, 1, nl=BN)
            if stride == 1 and out_channel == in_channel:
                l = l + shortcut
            return l

        with argscope([Conv2D, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope([Conv2D], use_bias=False):
            l = Conv2D('covn1', image, 32, 3, stride=2, nl=BNReLU)
            with tf.variable_scope('bottleneck1'):
                l = bottleneck_v2(l, out_channel=16, t=1, stride=1)

            with tf.variable_scope('bottleneck2'):
                for j in range(2):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channel=24, t=6, stride=2 if j == 0 else 1)

            with tf.variable_scope('bottleneck3'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channel=32, t=6, stride=2 if j == 0 else 1)
            
            with tf.variable_scope('bottleneck4'):
                for j in range(4):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channel=64, t=6, stride=2 if j == 0 else 1)
            
            with tf.variable_scope('bottleneck5'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channel=96, t=6, stride=1)
            
            with tf.variable_scope('bottleneck6'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channel=160, t=6, stride=2 if j== 0 else 1)
            with tf.variable_scope('bottleneck7'):
                l = bottleneck_v2(l, out_channel=320, t=6, stride=1)
            l = Conv2D('conv2', l, 1280, 1, nl=BNReLU)
            l = GlobalAvgPooling('gap', l)
            l = Dropout("dp", l, cfg.dropout)
            logits = FullyConnected('linear', l, cfg.class_num)
            
            return logits

def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            GoogleNetResize(crop_area_fraction=0.49),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((cfg.h, cfg.w)),
        ]
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, nr_tower, args):
    # batch = TOTAL_BATCH_SIZE // nr_tower
    batch = args.batch_size // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)
    callbacks = [
        ModelSaver(),
        HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: 1e-3),
                                     # lambda e, x: 4.5e-2 * 0.98 ** e),
        HumanHyperParamSetter('learning_rate'),
    ]
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=cfg.steps_per_epoch,
        max_epoch=cfg.max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default="0,1")
    parser.add_argument('--data', help='ILSVRC dataset dir', default='ILSVRC2012')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()

    if args.eval:
        batch = args.batch_size    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    elif args.flops:
        # manually build the graph with batch=1
        input_desc = [
            InputDesc(tf.float32, [1, cfg.h, cfg.w, 3], 'input'),
            InputDesc(tf.int32, [1], 'label')
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
        logger.set_logger_dir(
            os.path.join('train_log', 'mobilenetv2'))

        nr_tower = max(get_nr_gpu(), 1)
        config = get_config(model, nr_tower, args)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
