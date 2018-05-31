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

from utils.nms_wrapper import nms

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

class Box():
    def __init__(self, p1, p2, p3, p4, mode='XYWH'):
        if mode == 'XYWH':
            # parameters: center_x, center_y, width, height
            self.x = p1
            self.y = p2
            self.w = p3
            self.h = p4
        if mode == "XYXY":
            # parameters: xmin, ymin, xmax, ymax
            self.x = (p1 + p3) / 2
            self.y = (p2 + p4) / 2
            self.w = p3 - p1
            self.h = p4 - p2

def encode_box(gt_box, anchor_box):
    cx = (gt_box.x - anchor_box.x) / anchor_box.w / cfg.prior_scaling[0]
    cy = (gt_box.y - anchor_box.y) / anchor_box.h / cfg.prior_scaling[0]
    w = np.log(gt_box.w / anchor_box.w) / cfg.prior_scaling[1]
    h = np.log(gt_box.h / anchor_box.h) / cfg.prior_scaling[1]
    return np.asarray([cx, cy, w, h])

def decode_box(loc_pred, anchor):
    # cx, cy, w, h = loc_pred
    # box_cx = cx * anchor_box.w * cfg.prior_scaling[0] + anchor_box.x
    # box_cy = cy * anchor_box.h * cfg.prior_scaling[1] + anchor_box.y
    # box_w = np.exp(w * cfg.prior_scaling[2]) * anchor_box.w
    # box_h = np.exp(h * cfg.prior_scaling[3]) * anchor_box.h
    # import pdb
    # pdb.set_trace() 
    decoded_loc = np.concatenate((anchor[:, :2] + loc_pred[:, :2] * cfg.prior_scaling[0] * anchor[:, 2:], 
                                  anchor[:, 2:] * np.exp(loc_pred[:, 2:] * cfg.prior_scaling[1])), 1)
    return decoded_loc
    # return Box(box_cx, box_cy, box_w, box_h)

def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(box1, box2):
    i = box_intersection(box1, box2)
    u = box1.w * box1.h + box2.w * box2.h - i
    return u

def box_iou(box1, box2):
    return box_intersection(box1, box2) / box_union(box1, box2)

def non_maximum_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    conf = boxes[:,4]
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        union = area[idxs[:last]] + area[idxs[last]] - intersection
 
        # compute the ratio of overlap
        # overlap = (w * h) / area[idxs[:last]]
        overlap = intersection / union
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("float")
    return pick

def postprocess(predictions, image_path=None, image_shape=None, det_th=None):
    # t = float(time.time())
    # print("1: %s" % time.time())
    if image_path != None:
        ori_image = cv2.imread(image_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image_shape = ori_image.shape
    ori_height = image_shape[0]
    ori_width = image_shape[1]

    [loc_pred, cls_pred] = predictions

    _, box_n, klass_num = cls_pred.shape

    width_rate = ori_width / float(cfg.img_w)
    height_rate = ori_height / float(cfg.img_h)

    decoded_loc = decode_box(loc_pred[0], cfg.all_anchors[:, :4])
    ori_wh = np.array([ori_width, ori_height])
    # import pdb
    # pdb.set_trace()
    xy_min = (decoded_loc[:, :2] - decoded_loc[:, 2:]/2) * ori_wh
    xy_max = (decoded_loc[:, :2] + decoded_loc[:, 2:]/2) * ori_wh
    trun_xyxy = np.concatenate((np.maximum(xy_min, np.zeros(xy_min.shape)), np.minimum(xy_max, np.ones(xy_max.shape)*ori_wh)), 1)
    boxes = {}
    # t2 = float(time.time())
    # print("2: %f" % (float(time.time()) - t))
    # print(box_n)
#     for n in range(box_n):
#         # t2 = float(time.time())
# 
#         selected_klass = np.where(cls_pred[0, n] > (det_th or cfg.det_th))[0].tolist()
#         # the 0th class in prediction is the background
#         selected_klass.remove(0) if 0 in selected_klass else None
#         if len(selected_klass) == 0:
#             continue
#         # klass = np.argmax(cls_pred[0, n, 1:]) + 1
#         # if cls_pred[0, n, klass] < (det_th or cfg.det_th):
#         #     continue
# 
#         # the class index in config file is 0-based
# #        anchor_box = Box(*cfg.all_anchors[n][:4])
# #        pred_box = decode_box(loc_pred[0, n], anchor_box)
# 
#         # t0 = float(time.time())
#         # print('2_0: %f' % (t0 - t2))
# 
# #         xmin = float(pred_box.x - pred_box.w / 2) * ori_width
# #         ymin = float(pred_box.y - pred_box.h / 2) * ori_height
# #         xmax = float(pred_box.x + pred_box.w / 2) * ori_width
# #         ymax = float(pred_box.y + pred_box.h / 2) * ori_height
# #         xmin = np.max([xmin, 0])
# #         ymin = np.max([ymin, 0])
# #         xmax = np.min([xmax, ori_width])
# #         ymax = np.min([ymax, ori_height])
# 
#         # tx = time.time()
#         # print('2_1: %f' % (float(tx) - t0))
# 
# 
#         for klass in selected_klass:
#             box = list(trun_xyxy[n])
#             klass_name = cfg.classes_name[klass - 1]
#             if klass_name not in boxes.keys():
#                 boxes[klass_name] = []
# 
# #             box = [xmin, ymin, xmax, ymax, cls_pred[0, n, klass]]
#             box.append(cls_pred[0, n, klass])
#             boxes[klass_name].append(box)
# 
#         # print('2_2: %f' % (float(time.time()) - tx))
# 
#     # t3 = float(time.time())
#     # print("3: %f" % (float(time.time()) - t2))
#     # do non-maximum-suppresion
#     # import pdb
#     # pdb.set_trace()
#     for klass_name, k_boxes in boxes.items():
#         boxes[klass_name] = np.asarray(k_boxes, dtype=np.float32)
    cls_pred = cls_pred[0]
    for j in range(1, cfg.class_num + 1):
        inds = np.where(cls_pred[:, j] > (det_th or cfg.det_th))[0]
        klass_name = cfg.classes_name[j - 1]
        if len(inds) == 0:
            boxes[klass_name] = np.empty([0, 5], dtype=np.float32)
            continue

        c_bboxes = trun_xyxy[inds]
        c_pred = cls_pred[inds,j]
        c_dets = np.hstack((c_bboxes, c_pred[:, np.newaxis])).astype(np.float32)
        boxes[klass_name] = c_dets

    nms_boxes = {}
    if cfg.nms == True:
        for klass_name, k_boxes in boxes.items():
            keep = nms(k_boxes, cfg.nms_th, force_cpu=False)
            # k_boxes = non_maximum_suppression(k_boxes, cfg.nms_th)
            # keep_2 = non_maximum_suppression(k_boxes, cfg.nms_th)
            nms_boxes[klass_name] = k_boxes[keep]
    else:
        nms_boxes = boxes
    
    # print("4: %f" % (float(time.time()) - t3))
    return nms_boxes
