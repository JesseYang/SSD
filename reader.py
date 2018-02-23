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
    from .utils import Box, box_iou, encode_box
except Exception:
    from cfgs.config import cfg
    from utils import Box, box_iou, encode_box

from tensorpack import *

SAVE_DIR = 'input_images'

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def cal_overlap(bboxes, crop_box):
    # import pdb
    # pdb.set_trace()
    inter = intersect(bboxes, crop_box)
    area_bboxes = ((bboxes[:, 2] - bboxes[:, 0]) *
                   (bboxes[:, 3] - bboxes[:, 1]))
    return inter / area_bboxes

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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union


class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, random_crop, save_img=False):
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

    def size(self):
        return len(self.imglist)

    def generate_sample(self, idx, image_height, image_width):
        hflip = False if self.flip == False else (random.random() > 0.5)
        line = self.imglist[idx]

        grid_h = int(image_height / 32)
        grid_w = int(image_width / 32)

        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])
        s = image.shape
        h, w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            misc.imsave(os.path.join(SAVE_DIR, "%d_with_box.jpg" % idx), ori_img_with_box)

        if hflip:
            image = cv2.flip(image, flipCode=1)
            boxes = boxes.copy()
            boxes[:, 0::2] = w - boxes[:, 2::-2]

        expand = 0
        if self.random_crop:
            # expand img
            expand = np.random.randint(2)
            if expand == 1:
                ratio = np.random.uniform(1, 4)
                left = np.random.uniform(0, w * ratio - w)
                top = np.random.uniform(0, h * ratio - h)
                expand_image = np.zeros(
                    (int(h * ratio), int(w * ratio), c),
                    dtype=image.dtype)

                img_mean = np.mean(image, axis=(0,1))
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
                [0.1, np.inf],
                [0.3, np.inf],
                [0.7, np.inf],
                [0.9, np.inf],
                [-np.inf, np.inf]]
            mode_idx = np.random.randint(len(sample_options))
            mode = sample_options[mode_idx]
            crop = False
            if mode != None:
                min_iou, max_iou = mode
                for _ in range(50):
                    current_image = image
                    crop_w = np.random.uniform(0.3 * w, w)
                    crop_h = np.random.uniform(0.3 * h, h)

                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2:
                        continue

                    left = np.random.uniform(w - crop_w)
                    top = np.random.uniform(h - crop_h)

                    rect = np.array([int(left), int(top), int(left + crop_w), int(top + crop_h)])

                    # overlap = jaccard_numpy(boxes, rect)
                    overlap = cal_overlap(boxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if (overlap.min() < min_iou) or (overlap.max() > max_iou):
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
                    box_num = boxes.shape[0]
                    s = image.shape
                    h, w, _ = image.shape



        img_with_box = np.copy(image) if self.save_img else None
        image = cv2.resize(image, (image_width, image_height))

        anchor_iou = np.zeros((cfg.tot_anchor_num, ))
        # the backgound class is the 0th class
        anchor_cls = np.zeros((cfg.tot_anchor_num, )).astype(int)
        anchor_loc = np.zeros((cfg.tot_anchor_num, 4))

        gt_box_num = 0
        gt_box_coord = np.zeros((cfg.max_gt_box_shown, 4))
        for box_idx in range(box_num):
            class_num = class_ary[box_idx]
            if self.save_img:
                cv2.rectangle(img_with_box,
                              (int(boxes[box_idx, 0]), int(boxes[box_idx, 1])),
                              (int(boxes[box_idx, 2]), int(boxes[box_idx, 3])),
                              self.colors[class_num % len(self.colors)],
                              3)

            xmin, xmax = boxes[box_idx, 0::2] / w
            ymin, ymax = boxes[box_idx, 1::2] / h

            if gt_box_num < cfg.max_gt_box_shown:
                gt_box_coord[gt_box_num] = np.asarray([ymin, xmin, ymax, xmax])
                gt_box_num += 1

            gt_box = Box(xmin, ymin, xmax, ymax, mode='XYXY')

            gt_box_a = gt_box.w * gt_box.h

            for anchor_idx, anchor in enumerate(cfg.all_anchors):
                if gt_box_a > anchor[4] or gt_box_a < anchor[5]:
                    continue
                if np.abs(gt_box.x - anchor[0]) > min(gt_box.w, anchor[2]) / 2:
                    continue
                if np.abs(gt_box.y - anchor[1]) > min(gt_box.h, anchor[3]) / 2:
                    continue
                anchor_box = Box(*anchor[:4])
                iou = box_iou(gt_box, anchor_box)
                if iou >= cfg.iou_th and iou > anchor_iou[anchor_idx]:
                    # the 0th class is the background, thus other classes' number should be pushed back 1
                    anchor_cls[anchor_idx] = class_num + 1
                    anchor_loc[anchor_idx] = encode_box(gt_box, anchor_box)
                if iou > anchor_iou[anchor_idx]:
                    anchor_iou[anchor_idx] = iou

        anchor_neg_mask = anchor_iou < cfg.neg_iou_th

        if self.save_img:
            misc.imsave(os.path.join(SAVE_DIR, "%d_with_box_aug_%d_%d_%d.jpg" % (idx, expand, mode_idx, int(crop))), img_with_box)

        return [image, gt_box_coord, anchor_cls, anchor_neg_mask, anchor_loc, np.asarray(s)]

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
    df = Data('temp.txt', shuffle=False, flip=True, random_crop=True, save_img=True)
    df.reset_state()

    g = df.get_data()
    for i in range(256):
        next(g)

    # for idx in range(100):
    #     if idx % 10 == 0:
    #         print(time.time())
    #     g = df.get_data()
    #     pb = next(g)
