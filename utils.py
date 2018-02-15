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
    cx = (gt_box.x - anchor_box.x) / anchor_box.w
    cy = (gt_box.y - anchor_box.y) / anchor_box.h
    w = np.log(gt_box.w / anchor_box.w)
    h = np.log(gt_box.h / anchor_box.h)
    return np.asarray([cx, cy, w, h])

def decode_box(loc_pred, anchor_box):
    cx, cy, w, h = loc_pred
    box_cx = cx * anchor_box.w + anchor_box.x
    box_cy = cy * anchor_box.h + anchor_box.y
    box_w = np.exp(w) * anchor_box.w
    box_h = np.exp(h) * anchor_box.h
    return Box(box_cx, box_cy, box_w, box_h)

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
    conf = boxes[:,0]
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
 
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
    return boxes[pick].astype("float")

def postprocess(predictions, image_path=None, image_shape=None, det_th=None):
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

    boxes = {}
    for n in range(box_n):
        klass = np.argmax(cls_pred[0, n])
        # skip the background predictions
        if klass == cfg.class_num:
            continue
        pred_box = decode_box(loc_pred[0, n], anchor_box[0, n]):



        xmin = float(pred_box.x - pred_box.w / 2) * ori_width
        ymin = float(pred_box.y - pred_box.h / 2) * ori_height
        xmax = float(pred_box.x + pred_box.w / 2) * ori_width
        ymax = float(pred_box.y + pred_box.h / 2) * ori_height
        xmin = np.max([xmin, 0])
        ymin = np.max([ymin, 0])
        xmax = np.min([xmax, ori_width])
        ymax = np.min([ymax, ori_height])

        klass_name = cfg.classes_name[klass]
        if klass_name not in boxes.keys():
            boxes[klass_name] = []

        box = [cls_pred[0, n, klass], xmin, ymin, xmax, ymax]

        boxes[klass].append(box)

    # do non-maximum-suppresion
    nms_boxes = {}
    if cfg.nms == True:
        for klass, k_boxes in boxes.items():
            k_boxes = non_maximum_suppression(k_boxes, cfg.nms_th)
            nms_boxes[klass] = k_boxes
    else:
        nms_boxes = boxes
    
    return nms_boxes
