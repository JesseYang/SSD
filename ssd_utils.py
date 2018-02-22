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
import time

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

    def __init__(self, data_format="NHWC"):
        super(SSDModel, self).__init__()
        self.data_format = data_format

    def ssd_multibox_layer(self, feature_idx, feature):
        anchor_sizes = cfg.anchor_sizes[feature_idx]
        anchor_ratios = cfg.anchor_ratios[feature_idx]

        anchor_num = len(anchor_sizes) + len(anchor_ratios)

        if self.data_format == 'NCHW':
            h = int(feature.get_shape()[2])
            w = int(feature.get_shape()[3])
        else:
            h = int(feature.get_shape()[1])
            w = int(feature.get_shape()[2])

        W_init = tf.contrib.layers.xavier_initializer()
        # location
        loc_pred_num = anchor_num * 4
        loc_pred = Conv2D('conv_loc', feature, loc_pred_num, 3, kernel_initializer=W_init)
        if self.data_format == 'NCHW':
            loc_pred = tf.transpose(loc_pred, [0, 2, 3, 1])
        # loc_pred = tf.reshape(loc_pred, [-1, h, w, anchor_num, 4])
        loc_pred = tf.reshape(loc_pred, [-1, h * w * anchor_num, 4])

        # class prediction
        cls_pred_num = anchor_num * (cfg.class_num + 1)
        cls_pred = Conv2D('conv_cls', feature, cls_pred_num, 3, kernel_initializer=W_init)
        if self.data_format == 'NCHW':
            loc_pred = tf.transpose(cls_pred, [0, 2, 3, 1])
        # cls_pred = tf.reshape(cls_pred, [-1, h, w, anchor_num, (cfg.class_num + 1)])
        cls_pred = tf.reshape(cls_pred, [-1, h * w * anchor_num, (cfg.class_num + 1)])

        return loc_pred, cls_pred

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.img_h, cfg.img_w, 3], 'input'),
                InputDesc(tf.float32, [None, cfg.max_gt_box_shown, 4], 'gt_bboxes'),
                InputDesc(tf.int32, [None, cfg.tot_anchor_num], 'conf_label'),
                InputDesc(tf.bool, [None, cfg.tot_anchor_num], 'neg_mask'),
                InputDesc(tf.float32, [None, cfg.tot_anchor_num, 4], 'loc_label'),
                InputDesc(tf.float32, [None, 3], 'ori_shape'),
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
        # image, gt_boxes_loc, gt_boxes_label = inputs
        image, gt_bbox, conf_label, neg_mask, loc_label, ori_shape = inputs
        self.batch_size = tf.shape(image)[0]

        image = tf.cast(image, tf.float32) * (1.0 / 255)

        image_with_bbox = tf.image.draw_bounding_boxes(image, gt_bbox)
        tf.summary.image('input-image', image_with_bbox, max_outputs=3)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        image = tf.identity(image, name='network_input')

        features = self.get_logits(image)

        loc_pred_list = []
        cls_pred_list = []
        for feature_idx, feature in enumerate(features):
            with tf.variable_scope('feature_layer_%d' % feature_idx):
                loc_pred, cls_pred = self.ssd_multibox_layer(feature_idx, feature)
            loc_pred_list.append(loc_pred)
            cls_pred_list.append(cls_pred)

        loc_pred = tf.concat(loc_pred_list, axis=1, name='loc_pred')
        cls_pred = tf.concat(cls_pred_list, axis=1)
        predictions = tf.nn.softmax(cls_pred, name='cls_pred')

        # the loss part of SSD
        nr_pos = tf.stop_gradient(tf.count_nonzero(conf_label, dtype=tf.int32))
        nr_pos = tf.identity(nr_pos, name='nr_pos')
        # location loss, the last class is the background class
        pos_mask = tf.stop_gradient(tf.not_equal(conf_label, 0))
        # neg_mask = tf.stop_gradient(tf.equal(conf_label, 0))
        loc_mask_label = tf.boolean_mask(loc_label, pos_mask)
        loc_mask_label = tf.identity(loc_mask_label, name='loc_mask_label')
        loc_mask_pred = tf.boolean_mask(loc_pred, pos_mask)
        loc_mask_pred = tf.identity(loc_mask_pred, name='loc_mask_pred')
        loc_loss = tf.losses.huber_loss(loc_mask_label, loc_mask_pred, reduction=tf.losses.Reduction.SUM)
        loc_loss = tf.identity(loc_loss, 'tot_loc_loss')
        # confidence loss
        if cfg.hard_sample_mining:

            dtype = cls_pred.dtype
            fpmask = tf.cast(pos_mask, dtype)
            fnmask = tf.cast(neg_mask, dtype)
            no_classes = tf.cast(pos_mask, tf.int32)

            neg_predictions = tf.where(neg_mask,
                                       predictions[:, :, 0],
                                       1. - fnmask)

            neg_pred_flat = tf.reshape(neg_predictions, [-1])

            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            nr_neg = tf.cast(cfg.neg_ratio * nr_pos, tf.int32) + self.batch_size
            nr_neg = tf.minimum(nr_neg, max_neg_entries)

            val, idxes = tf.nn.top_k(-neg_pred_flat, k=nr_neg)
            max_hard_pred = -val[-1]
            neg_mask = tf.logical_and(neg_mask, neg_predictions <= max_hard_pred)
            fnmask = tf.cast(neg_mask, dtype)

            pos_conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred, labels=conf_label)
            pos_conf_loss = tf.reduce_sum(pos_conf_loss * fpmask, name='pos_conf_loss')

            neg_conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred, labels=no_classes)
            neg_conf_loss = tf.reduce_sum(neg_conf_loss * fnmask, name='neg_conf_loss')

            conf_loss = pos_conf_loss + neg_conf_loss
            # add_moving_summary(pos_conf_loss, neg_conf_loss)
        else:
            conf_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred, labels=conf_label))
        # cost with weight decay
        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)
        loc_loss = tf.truediv(loc_loss, tf.to_float(nr_pos), name='loc_loss')
        conf_loss = tf.truediv(conf_loss, tf.to_float(nr_pos), name='conf_loss')
        # loc_loss = tf.truediv(loc_loss, tf.to_float(self.batch_size), name='loc_loss')
        # conf_loss = tf.truediv(conf_loss, tf.to_float(self.batch_size), name='conf_loss')

        loss = tf.add_n([loc_loss * cfg.alpha, conf_loss], name='loss')
        add_moving_summary(loc_loss, conf_loss, loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-3, summary=True)
        # return tf.train.AdamOptimizer(lr,
        #                               beta1=0.9,
        #                               beta2=0.999,
        #                               epsilon=1.0)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

class CalMAP(Inferencer):
    def __init__(self, test_path):
        self.names = ["loc_pred", "cls_pred", "ori_shape", "loss"]
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
        self.loss.append(output[3])
        output = output[0:3]
        for i in range(output[0].shape[0]):
            # for each ele in the batch
            image_path = self.image_path_list[self.cur_image_idx]
            self.cur_image_idx += 1
            image_id = os.path.basename(image_path).split('.')[0] if cfg.gt_format == "voc" else image_path

            cur_output = [ele[i] for ele in output]
            predictions = [np.expand_dims(ele, axis=0) for ele in cur_output[0:2]]
            image_shape = cur_output[2]


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
        return { "mAP": mAP }


def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, flip=isTrain, random_crop=isTrain)

    if isTrain:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(32, clip=False),
                 imgaug.Contrast((0.5, 1.5), clip=False),
                 imgaug.Saturation(0.5),
                 imgaug.Hue((0, 0.2*180), rgb=True)]),
            imgaug.Clip(),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds


def get_config(args, model):
    if args.gpu != None:
        NR_GPU = len(args.gpu.split(','))
        batch_size = int(args.batch_size) // NR_GPU
    else:
        batch_size = int(args.batch_size)

    ds_train = get_data('train', batch_size)
    ds_test = get_data('test', batch_size)

    callbacks = [
      ModelSaver(),
      PeriodicTrigger(InferenceRunner(ds_test,
                                      ScalarStats(['conf_loss', 'loc_loss', 'loss'])),
                      every_k_epochs=3),
      ScheduledHyperParamSetter('learning_rate',
                                cfg.lr_schedule),
      # HyperParamSetterWithFunc('learning_rate',
      #                          lambda e, x: 1e-3 * 0.98 ** e),
      HumanHyperParamSetter('learning_rate'),
    ]
    if cfg.mAP == True:
        callbacks.append(EnableCallbackIf(PeriodicTrigger(InferenceRunner(ds_test,
                                                                         [CalMAP(cfg.test_list)]),
                                          every_k_epochs=3),
                         lambda x : x.epoch_num >= 10)),

    if args.debug:
      callbacks.append(HookToCallback(tf_debug.LocalCLIDebugHook()))
    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=cfg.train_sample_num // args.batch_size,
        max_epoch=args.itr // (cfg.train_sample_num // args.batch_size),
    )
