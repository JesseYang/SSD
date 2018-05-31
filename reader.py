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
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
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

    def size(self):
        return len(self.imglist)

    def generate_sample(self, idx, image_height, image_width):
        hflip = False if self.flip == False else (random.random() > 0.5)
        line = self.imglist[idx]

        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])
        s = image.shape
        h, w, c = image.shape
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box_num = (len(record) - 1) // 5
        boxes = np.zeros((box_num, 4))
        class_ary = np.zeros((box_num)).astype(int)
        box_idx = 0
        i = 1
        while i < len(record):
            # for each ground truth box
            for coord_idx in range(4):
                boxes[box_idx, coord_idx] = record[i + coord_idx]
            class_ary[box_idx] = int(record[i + 4])
            box_idx += 1
            i += 5

        ori_img_with_box = image.copy()
        if self.save_img:
            for box_idx in range(box_num):
                cv2.rectangle(ori_img_with_box,
                              (int(boxes[box_idx, 0]), int(boxes[box_idx, 1])),
                              (int(boxes[box_idx, 2]), int(boxes[box_idx, 3])),
                              self.colors[class_ary[box_idx] % len(self.colors)],
                              3)
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_with_box.jpg" % idx), ori_img_with_box)

        if hflip:
            image = cv2.flip(image, flipCode=1)
            boxes = boxes.copy()
            boxes[:, 0::2] = w - boxes[:, 2::-2]

        ori_image = image.copy()
        ori_boxes = boxes.copy()

        expand = 0
        if self.random_crop:
            # expand img
            expand = np.random.randint(2) if self.random_expand else 0
            if expand == 1:
                ratio = np.random.uniform(1, 4)
                left = np.random.uniform(0, w * ratio - w)
                top = np.random.uniform(0, h * ratio - h)
                expand_image = np.zeros(
                    (int(h * ratio), int(w * ratio), c),
                    dtype=image.dtype)

                img_mean = np.asarray([104, 117, 123])
                expand_image[:, :, :] = img_mean
                expand_image[int(top):int(top + h),
                             int(left):int(left + w)] = image
                image = expand_image
                s = image.shape
                h, w, _ = image.shape

                boxes = boxes.copy()
                boxes[:, :2] += (int(left), int(top))
                boxes[:, 2:] += (int(left), int(top))

            sample_options = [
                None,   # use entire original input image
                [0.1, 1.0],
                [0.3, 1.0],
                [0.5, 1.0],
                [0.7, 1.0],
                [0.9, 1.0],
                [0.0, 1.0]]
            mode_idx = np.random.randint(len(sample_options))
            mode = sample_options[mode_idx]
            crop = False
            if mode != None:
                min_iou, max_iou = mode
                for _ in range(50):
                    current_image = image

                    scale = np.random.uniform(0.3, 1.0)
                    aspect_ratio = np.random.uniform(0.5, 2.0)

                    aspect_ratio = np.maximum(aspect_ratio, scale ** 2)
                    aspect_ratio = np.minimum(aspect_ratio, 1 / scale ** 2)

                    crop_w = scale * np.sqrt(aspect_ratio) * w
                    crop_h = scale / np.sqrt(aspect_ratio) * h

                    left = np.random.uniform(w - crop_w)
                    top = np.random.uniform(h - crop_h)

                    rect = np.array([int(left), int(top), int(left + crop_w), int(top + crop_h)])

                    overlap = jaccard_numpy(boxes, rect)

                    satisfy = (overlap > min_iou) * (overlap < max_iou)

                    # is min and max overlap constraint satisfied? if not try again
                    if not satisfy.any():
                        continue

                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # take only matching gt labels
                    current_class_ary = class_ary[mask]

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                      rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                      rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    crop = True
                    break

                if crop == True:
                    image = current_image
                    boxes = current_boxes
                    class_ary = current_class_ary
                else:
                    image = ori_image
                    boxes = ori_boxes
                box_num = boxes.shape[0]
                s = image.shape
        h, w, _ = image.shape

        boxes[:,0::2] = boxes[:,0::2] / w
        boxes[:,1::2] = boxes[:,1::2] / h

        img_with_box = np.copy(image) if self.save_img else None
        if self.random_inter:
            inter_modes = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
            inter_mode_idx = np.random.randint(5)
            image = cv2.resize(image, (image_width, image_height), inter_modes[inter_mode_idx])
        else:
            image = cv2.resize(image, (image_width, image_height))

        # only three variables are used in the following code:
        # img: 300 x 300 x 3 numpy array
        # boxes: N x 4 numpy array, N is boxes number, coordinates for each box
        # class_ary: N numpy array, N is boxes number, class for each box
        # these variables can be generated by the pytorch version dataset, and the data part can be switched to pytorch version
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
        if box_num >= cfg.max_gt_box_shown:
            gt_box_coord = boxes[:cfg.max_gt_box_shown][:,np.asarray([1, 0, 3, 2])]
        else:
            gt_box_coord = np.zeros((cfg.max_gt_box_shown, 4))
            gt_box_coord[:box_num] = boxes[:,np.asarray([1, 0, 3, 2])]

        if self.save_img:
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_with_box_aug_%d_%d_%d.jpg" % (idx, expand, mode_idx, int(crop))), img_with_box)

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
    df = Data('temp.txt', shuffle=True, flip=True, random_crop=True, random_expand=True, random_inter=True, save_img=True)
    df.reset_state()

    g = df.get_data()
    for i in range(256):
        next(g)

    # for idx in range(100):
    #     if idx % 10 == 0:
    #         print(time.time())
    #     g = df.get_data()
    #     pb = next(g)
