import os, sys, shutil
import math
import time
import pickle
import numpy as np
import random
from scipy import misc
import six
import random
from six.moves import urllib, range
import copy
import logging
import cv2
import json
import uuid

from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc

try:
    from .cfgs.config import cfg
    from .utils_bak import Box, box_iou, encode_box
except Exception:
    from cfgs.config import cfg
    from utils_bak import Box, box_iou, encode_box

from tensorpack import *

SAVE_DIR = 'input_images'


def encode(gt_box):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (gt_box[:, :2] + gt_box[:, 2:]) / 2 - cfg.all_anchors[:, :2]
    # encode variance
    g_cxcy /= (cfg.prior_scaling[0] * cfg.all_anchors[:, 2:])
    # match wh / prior wh
    g_wh = (gt_box[:, 2:] - gt_box[:, :2]) / cfg.all_anchors[:, 2:]
    g_wh = np.log(g_wh) / cfg.prior_scaling[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate([
        cfg.all_anchors[:, :2] + loc[:, :2] * cfg.prior_scaling[0] * cfg.all_anchors[:, 2:],
        cfg.all_anchors[:, 2:] * np.exp(loc[:, 2:] * cfg.prior_scaling[1])], 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: center-size default boxes from priorbox layers.
    Return:
        boxes: Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,     # xmin, ymin
                           boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax

def intersect(box_a, box_b):
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    max_xy_a = np.repeat(np.expand_dims(box_a[:, 2:], axis=1), num_b, axis=1)
    max_xy_b = np.repeat(np.expand_dims(box_b[:, 2:], axis=0), num_a, axis=0)
    max_xy = np.minimum(max_xy_a, max_xy_b)

    min_xy_a = np.repeat(np.expand_dims(box_a[:, :2], axis=1), num_b, axis=1)
    min_xy_b = np.repeat(np.expand_dims(box_b[:, :2], axis=0), num_a, axis=0)
    min_xy = np.maximum(min_xy_a, min_xy_b)


    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes_1, 4]
        box_b: Multiple bounding boxes, Shape: [num_boxes_2, 4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_b.shape[0]]
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    inter = intersect(box_a, box_b)

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_a = np.repeat(np.expand_dims(area_a, axis=1), num_b, axis=1)

    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])  # [A,B]
    area_b = np.repeat(np.expand_dims(area_b, axis=0), num_a, axis=0)

    union = area_a + area_b - inter
    return inter / union

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, random_crop, random_expand, random_inter, random_distort, save_img=False):
        self.filename_list = filename_list
        self.save_img = save_img

        if save_img == True:
            if os.path.isdir(SAVE_DIR):
                shutil.rmtree(SAVE_DIR)
            os.mkdir(SAVE_DIR)
            self.colors = [(255,0,0), (0,255,0), (0,0,255),
                           (255,255,0), (255,0,255), (0,255,255),
                           (122,0,0), (0,122,0), (0,0,122),
                           (122,122,0), (122,0,122), (0,122,122)]

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        self.random_crop = random_crop
        self.random_expand = random_expand
        self.random_inter = random_inter
        self.random_distort = random_distort

    def size(self):
        return len(self.imglist)

    def _crop(self, image, boxes, labels):
        height, width, _ = image.shape
    
        def matrix_iou(a, b):
            """
            return iou of a and b, numpy version for data augenmentation
            """
            lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
            rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        
            area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
            area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
            area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
            return area_i / (area_a[:, np.newaxis] + area_b - area_i)

        if len(boxes) == 0:
            return image, boxes, labels
    
        while True:
            mode = random.choice((
                None,
                (0.1, None),
                (0.3, None),
                (0.5, None),
                (0.7, None),
                (0.9, None),
                (None, None),
            ))
    
            if mode is None:
                return image, boxes, labels
    
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
    
            for _ in range(50):
                scale = random.uniform(0.3, 1.)
                min_ratio = max(0.5, scale * scale)
                max_ratio = min(2, 1. / scale / scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                w = int(scale * ratio * width)
                h = int((scale / ratio) * height)
    
                l = random.randrange(width - w)
                t = random.randrange(height - h)
                roi = np.array((l, t, l + w, t + h))
    
                iou = matrix_iou(boxes, roi[np.newaxis])
    
                if not (min_iou <= iou.min() and iou.max() <= max_iou):
                    continue
    
                image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
    
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                    .all(axis=1)
                boxes_t = boxes[mask].copy()
                labels_t = labels[mask].copy()
                if len(boxes_t) == 0:
                    continue
    
                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
                boxes_t[:, :2] -= roi[:2]
                boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
                boxes_t[:, 2:] -= roi[:2]
    
                return image_t, boxes_t, labels_t

        return image, boxes, labels


    def _distort(self, image):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp
    
        image = image.copy()
    
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
    
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
    
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
    
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
        return image

    def _expand(self, image, boxes, fill=(104, 117, 123), p=0.6):
        if random.random() > p:
            return image, boxes
    
        height, width, depth = image.shape
        for _ in range(50):
            scale = random.uniform(1, 4)
    
            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws < 1 or hs < 1:
                continue
            w = int(ws * width)
            h = int(hs * height)
    
            left = random.randint(0, w - width)
            top = random.randint(0, h - height)
    
            boxes_t = boxes.copy()
            boxes_t[:, :2] += (left, top)
            boxes_t[:, 2:] += (left, top)
    
            expand_image = np.empty(
                (h, w, depth),
                dtype=image.dtype)
            expand_image[:, :] = fill
            expand_image[top:top + height, left:left + width] = image
            image = expand_image
    
            return image, boxes_t

    def _mirror(self, image, boxes):
        _, width, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes

    def generate_sample(self, idx, image_height, image_width):

        line = self.imglist[idx]

        record = line.split(' ')
        anno = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])

        # extract annotation information
        box_num = len(anno) // 5
        boxes = np.zeros((box_num, 4))
        labels = np.zeros((box_num)).astype(int)
        for box_idx in range(box_num):
            for coord_idx in range(4):
                boxes[box_idx, coord_idx] = anno[box_idx * 5 + coord_idx]
            labels[box_idx] = int(anno[box_idx * 5 + 4])
            box_idx += 1

        if self.random_crop:
            image, boxes, labels = self._crop(image, boxes, labels)

        if self.random_distort:
            image = self._distort(image)

        if self.random_expand:
            image, boxes = self._expand(image, boxes)

        if self.flip == True:
            image, boxes = self._mirror(image, boxes)


        box_num = boxes.shape[0]

        h, w, _ = image.shape

        boxes[:,0::2] = boxes[:,0::2] / w
        boxes[:,1::2] = boxes[:,1::2] / h

        if self.random_inter:
            inter_modes = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
            inter_mode_idx = np.random.randint(5)
            image = cv2.resize(image, (image_width, image_height), inter_modes[inter_mode_idx])
        else:
            image = cv2.resize(image, (image_width, image_height))

        # only three variables are used in the following code:
        # image: 300 x 300 x 3 numpy array
        # boxes: N x 4 numpy array, N is boxes number, coordinates for each box
        # labels: N numpy array, N is boxes number, class for each box

        anchor_cls = np.zeros((cfg.tot_anchor_num, )).astype(int)
        anchor_loc = np.zeros((cfg.tot_anchor_num, 4))

        ious = jaccard_numpy(boxes, point_form(cfg.all_anchors))

        best_prior_overlap = np.max(ious, axis=1)
        best_prior_idx = np.argmax(ious, axis=1)

        # num_anchors, best gt for each anchor
        best_truth_overlap = np.max(ious, axis=0)
        best_truth_idx = np.argmax(ious, axis=0)

        # ensure best anchor box
        for anchor_idx in best_prior_idx:
            best_truth_overlap[anchor_idx] = 2

        # ensure every gt matches with its prior of max overlap
        for j in range(box_num):
            best_truth_idx[best_prior_idx[j]] = j

        anchor_cls = labels[best_truth_idx]
        anchor_cls[best_truth_overlap < cfg.threshold] = 0

        anchor_loc = boxes[best_truth_idx]
        anchor_loc = encode(anchor_loc)

        anchor_neg_mask = best_truth_overlap < cfg.neg_iou_th

        # gt_box_coord should be the format of (ymin, xmin, ymax, xmax)
        if box_num >= cfg.max_gt_box_shown:
            gt_box_coord = boxes[:cfg.max_gt_box_shown][:,np.asarray([1, 0, 3, 2])]
        else:
            gt_box_coord = np.zeros((cfg.max_gt_box_shown, 4))
            gt_box_coord[:box_num] = boxes[:,np.asarray([1, 0, 3, 2])]


        return [image, gt_box_coord, anchor_cls, anchor_neg_mask, anchor_loc, np.asarray([cfg.img_h, cfg.img_w, 3])]

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        image_height = cfg.img_h
        image_width = cfg.img_w
        for k in idxs:
            retval = self.generate_sample(k, image_height, image_width)
            if retval == None:
                continue
            yield retval

    def get_data_idx(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield k

    def reset_state(self):
        super(Data, self).reset_state()

def generate_gt_result(test_path, gt_dir="result_gt", overwrite=True):
    if overwrite == False and os.path.isdir(gt_dir):
        return
    # generate the ground truth files for calculation of average precision
    if overwrite == True and os.path.isdir(gt_dir):
        shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)


    with open(test_path) as f:
        content = f.readlines()

    gt_all = {}

    for line in content:
        record = line.split(' ')
        image_id = os.path.basename(record[0]).split('.')[0] if cfg.gt_format == "voc" else record[0]
        i = 1
        
        gt_cur_img = {}
        while i < len(record):
            class_num = int(record[i + 4])
            class_name = cfg.classes_name[class_num]
            
            if class_name not in gt_cur_img.keys():
                gt_cur_img[class_name] = []
            gt_cur_img[class_name].extend(record[i:i+4])
            
            i += 5
        
        for class_name, boxes in gt_cur_img.items():
            if class_name not in gt_all:
                gt_all[class_name] = []
            d = [image_id]
            d.extend(boxes)
            gt_all[class_name].append(d)
            

    for class_name in cfg.classes_name:
        if class_name in gt_all.keys():
            with open(os.path.join(gt_dir, class_name + ".txt"), 'w') as f:
                for line in gt_all[class_name]:
                    line = [str(ele) for ele in line]
                    f.write(' '.join(line) + '\n')

if __name__ == '__main__':
    # df = Data('voc_2007_train.txt', shuffle=False, flip=False, affine_trans=False)
    train_list = ["voc_2007_train.txt", "voc_2012_train.txt", "voc_2007_val.txt", "voc_2012_val.txt"]
    df = Data(train_list, shuffle=True, flip=True, random_crop=True, random_expand=True, random_inter=True, random_distort=True, save_img=True)
    df.reset_state()

    g = df.get_data()
    for i in range(256):
        next(g)

    # for idx in range(100):
    #     if idx % 10 == 0:
    #         print(time.time())
    #     g = df.get_data()
    #     pb = next(g)
