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
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

try:
    from .cfgs.config import cfg
    from .reader import Data, generate_gt_result
    from .evaluate import do_python_eval
    from .utils import postprocess
except Exception:
    from cfgs.config import cfg
    from reader import Data, generate_gt_result
    from evaluate import do_python_eval
    from utils import postprocess

class SSDModel(ModelDesc):

    def __init__(self, data_format="NHWC", multi_scale=False):
        super(SSDModel, self).__init__()
        self.data_format = data_format
        self.multi_scale = multi_scale

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.img_h, cfg.img_w, 3], 'input'),
                ]

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of cfg.img_hxcfg.img_w in ``self.data_format``
        Returns:
            cfg.img_h/32 x cfg.img_w/32 logits in ``self.data_format``
        """

    def _build_graph(self, inputs):
        image = inputs[0]
        self.batch_size = tf.shape(image)[0]

        tf.summary.image('input-image', image, max_outputs=3)

        image = tf.cast(image, tf.float32) * (1.0 / 255)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        image = tf.identity(image, name='network_input')

        logits = self.get_logits(image)

        # the loss part of SSD, confirm that logits is NCHW format
        # self.cost = tf.add_n([loss, wd_cost], name='cost')
        self.cost = tf.reduce_sum(logits, name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

class CalMAP(Inferencer):
    def __init__(self, test_path):
        self.names = ["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob", "ori_shape", "loss"]
        self.test_path = test_path
        self.gt_dir = "result_gt"
        if os.path.isdir(self.gt_dir):
            shutil.rmtree(self.gt_dir)

        self.pred_dir = "result_pred/"
        if os.path.isdir(self.pred_dir):
            shutil.rmtree(self.pred_dir)
        os.mkdir(self.pred_dir)

        with open(self.test_path) as f:
            content = f.readlines()

        self.image_path_list = []
        for line in content:
            self.image_path_list.append(line.split(' ')[0])

        self.cur_image_idx = 0

    def _get_fetches(self):
        return self.names

    def _before_inference(self):
        # if the "result_gt" dir does not exist, generate it from the data_set
        generate_gt_result(self.test_path, self.gt_dir, overwrite=False)
        self.results = { }
        self.loss = []
        self.cur_image_idx = 0

    def _on_fetches(self, output):
        self.loss.append(output[7])
        output = output[0:7]
        for i in range(output[0].shape[0]):
            # for each ele in the batch
            image_path = self.image_path_list[self.cur_image_idx]
            self.cur_image_idx += 1
            image_id = os.path.basename(image_path).split('.')[0] if cfg.gt_format == "voc" else image_path

            cur_output = [ele[i] for ele in output]
            predictions = [np.expand_dims(ele, axis=0) for ele in cur_output[0:6]]
            image_shape = cur_output[6]

            pred_results = postprocess(predictions, image_shape=image_shape)
            for class_name in pred_results.keys():
                if class_name not in self.results.keys():
                    self.results[class_name] = []
                for box in pred_results[class_name]:
                    record = [image_id]
                    record.extend(box)
                    record = [str(ele) for ele in record]
                    self.results[class_name].append(' '.join(record))

    def _after_inference(self):
        # write the result to file
        for class_name in self.results.keys():
            with open(os.path.join(self.pred_dir, class_name + ".txt"), 'wt') as f:
                for record in self.results[class_name]:
                    f.write(record + '\n')
        # calculate the mAP based on the predicted result and the ground truth
        mAP = do_python_eval(self.pred_dir)
        return { "mAP": mAP, "test_loss": np.mean(self.loss) }


def get_data(train_or_test, multi_scale, batch_size):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, flip=isTrain, affine_trans=isTrain, use_multi_scale=isTrain and multi_scale, period=batch_size*10)

    if isTrain:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain and multi_scale == False:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds


def get_config(args, model):
    if args.gpu != None:
        NR_GPU = len(args.gpu.split(','))
        batch_size = int(args.batch_size) // NR_GPU
    else:
        batch_size = int(args.batch_size)

    ds_train = get_data('train', args.multi_scale, batch_size)
    ds_test = get_data('test', False, batch_size)

    callbacks = [
      ModelSaver(),


      ScheduledHyperParamSetter('learning_rate',
                                cfg.lr_schedule),
      ScheduledHyperParamSetter('unseen_scale',
                                [(0, cfg.unseen_scale), (cfg.unseen_epochs, 0)]),
      HumanHyperParamSetter('learning_rate'),
    ]
    if cfg.mAP == True:
        callbacks.append(EnableCallbackIf(PeriodicTrigger(InferenceRunner(ds_test, [CalMAP(cfg.test_list)]), every_k_epochs=3),
                                          lambda x : x.epoch_num >= args.map_start_epoch))
    if args.debug:
      callbacks.append(HookToCallback(tf_debug.LocalCLIDebugHook()))
    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=3620,
        max_epoch=cfg.max_epoch,
    )
