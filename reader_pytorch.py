import os, sys, shutil
import time
import pickle
import numpy as np
import random
from scipy import misc
import six
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

rgb_std = (1, 1, 1)
img_dim = (300, 512)[0]
# rgb_means = (104, 117, 123)
rgb_means = (0, 0, 0)
p = (0.6, 0.2)[0]


class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, random_crop, random_expand, random_inter, save_img=False):
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

        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        self.train_dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, rgb_std, p), AnnotationTransform())

    def size(self):
        return len(self.imglist)

    def generate_sample(self, idx, image_height, image_width):

        image, target = self.train_dataset.__getitem__(idx)

        image = np.transpose(image.numpy(), (1, 2, 0))
        boxes = target[:,0:4]
        class_ary = target[:,-1]
        box_num = class_ary.shape[0]
        w = 300
        h = 300

        # only three variables are used in the following code:
        # image: 300 x 300 x 3 numpy array
        # boxes: N x 4 numpy array, N is boxes number, uniform coordinates for each box
        # class_ary: N numpy array, N is boxes number, class for each box, 1-indexed, 0 indicates background
        # these variables can be generated by the pytorch version dataset, and the data part can be switched to pytorch version


        _boxes = (boxes * 300).astype(int)
        image_show = np.copy(image)
        for box in _boxes:
            cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        cv2.imwrite('show.jpg', image_show)

        anchor_loc = np.zeros((cfg.tot_anchor_num, 4))
        anchor_cls = np.zeros((cfg.tot_anchor_num, )).astype(int)

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

        anchor_cls = class_ary[best_truth_idx]
        anchor_cls[best_truth_overlap < cfg.threshold] = 0

        anchor_loc = boxes[best_truth_idx]
        anchor_loc = encode(anchor_loc)

        anchor_neg_mask = best_truth_overlap < cfg.neg_iou_th

        # gt_box_coord should be the format of (ymin, xmin, ymax, xmax)
        if box_num >= cfg.tot_anchor_num:
            gt_box_coord = boxes[:cfg.tot_anchor_num][:,np.asarray([1, 0, 3, 2])]
        else:
            gt_box_coord = np.zeros((cfg.tot_anchor_num, 4))
            gt_box_coord[:box_num] = boxes[:,np.asarray([1, 0, 3, 2])]


        return [image, gt_box_coord, anchor_cls, anchor_neg_mask, anchor_loc, np.asarray([300,300, 3])]

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
    df = Data(train_list, shuffle=False, flip=False, random_crop=False, random_expand=False, random_inter=False, save_img=True)
    df.reset_state()

    g = df.get_data()
    for i in range(256):
        dp = next(g)
        import pdb
        pdb.set_trace()

    # for idx in range(100):
    #     if idx % 10 == 0:
    #         print(time.time())
    #     g = df.get_data()
    #     pb = next(g)
