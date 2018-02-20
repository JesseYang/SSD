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


class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, affine_trans):
        self.filename_list = filename_list

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        self.affine_trans = affine_trans

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

        if self.affine_trans:
            scale = np.random.uniform() / 10. + 1.
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            image = image[offy: (offy + h), offx: (offx + w)]

        if hflip:
            image = cv2.flip(image, flipCode=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_width, image_height))

        anchor_iou = np.zeros((cfg.tot_anchor_num, ))
        # the backgound class is the last class
        anchor_cls = np.zeros((cfg.tot_anchor_num, )).astype(int)
        anchor_loc = np.zeros((cfg.tot_anchor_num, 4))

        gt_box_num = 0
        gt_box_coord = np.zeros((cfg.max_gt_box_shown, 4))

        i = 1
        while i < len(record):
            # for each ground truth box
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            if self.affine_trans:
                box = np.asarray([xmin, ymin, xmax, ymax])
                box = box * scale
                box[0::2] -= offx
                box[1::2] -= offy
                xmin = np.maximum(0, box[0])
                ymin = np.maximum(1, box[1])
                xmax = np.minimum(w - 1, box[2])
                ymax = np.minimum(h - 1, box[3])
            if hflip:
                xmin = w - 1 - xmin
                xmax = w - 1 - xmax
                tmp = xmin
                xmin = xmax
                xmax = tmp
            class_num = int(record[i + 4])
            i += 5

            xmin = xmin / w
            xmax = xmax / w
            ymin = ymin / h
            ymax = ymax / h

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

        return [image, gt_box_coord, anchor_cls, anchor_neg_mask, anchor_loc, np.asarray(s)]

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        image_height = cfg.img_h
        image_width = cfg.img_w
        for k in idxs:
            yield self.generate_sample(k, image_height, image_width)

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
    df = Data('voc_2007_train.txt', shuffle=False, flip=False, affine_trans=False)
    df.reset_state()

    g = df.get_data()
    pb = next(g)

    import pdb
    pdb.set_trace()

    # for idx in range(100):
    #     if idx % 10 == 0:
    #         print(time.time())
    #     g = df.get_data()
    #     pb = next(g)
