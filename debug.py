#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

from utils import *

try:
    from .cfgs.config import cfg
    from .utils import postprocess
except Exception:
    from cfgs.config import cfg
    from utils import postprocess

try:
    from .vgg_ssd import VGGSSD
except Exception:
    from vgg_ssd import VGGSSD

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = VGGSSD()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "conf_label", "loc_label", "ori_shape"],
                                   output_names=["loc_pred", "cls_pred", "loc_mask_label", "loc_mask_pred", "tot_loc_loss", "nr_pos", "pos_conf_loss", "neg_conf_loss"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def draw_result(image, boxes):
    colors = [(255,0,0), (0,255,0), (0,0,255),
              (255,255,0), (255,0,255), (0,255,255),
              (122,0,0), (0,122,0), (0,0,122),
              (122,122,0), (122,0,122), (0,122,122)]

    text_colors = [(0,255,255), (255,0,255), (255,255,0),
                  (0,0,255), (0,255,0), (255,0,0),
                  (0,122,122), (122,0,122), (122,122,0),
                  (0,0,122), (0,122,0), (122,0,0)]

    image_result = np.copy(image)
    k_idx = 0
    for klass, k_boxes in boxes.items():
        for k_box in k_boxes:

            [conf, xmin, ymin, xmax, ymax] = k_box

            label_height = 14
            label_width = len(klass) * 10
 
            cv2.rectangle(image_result,
                          (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)),
                          colors[k_idx % len(colors)],
                          3)
            cv2.rectangle(image_result,
                          (int(xmin) - 2, int(ymin) - label_height),
                          (int(xmin) + label_width, int(ymin)),
                          colors[k_idx % len(colors)],
                          -1)
            cv2.putText(image_result,
                        klass,
                        (int(xmin), int(ymin) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_colors[k_idx % len(text_colors)])
        k_idx += 1

    return image_result

def predict_image(predict_func, det_th):

    file_name = 'voc_2007_train.txt'
    f = open(file_name, 'r')
    line = f.readlines()[0].strip()
    record = line.split(' ')
    input_path = record[0]

    ori_image = cv2.imread(input_path)
    s = ori_image.shape
    h, w, _ = ori_image.shape
    cvt_clr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(cvt_clr_image, (cfg.img_w, cfg.img_h))

    record[1:] = [float(num) for num in record[1:]]

    xmin = record[1]
    ymin = record[2]
    xmax = record[3]
    ymax = record[4]

    class_num = int(record[5])

    anchor_iou = np.zeros((cfg.tot_anchor_num, ))
    anchor_cls = np.zeros((cfg.tot_anchor_num, )).astype(int)
    anchor_loc = np.zeros((cfg.tot_anchor_num, 4))

    xmin = xmin / w
    xmax = xmax / w
    ymin = ymin / h
    ymax = ymax / h

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
            anchor_iou[anchor_idx] = iou
            anchor_cls[anchor_idx] = class_num + 1
            anchor_loc[anchor_idx] = encode_box(gt_box, anchor_box)


    image = np.expand_dims(image, axis=0)
    anchor_cls = np.expand_dims(anchor_cls, axis=0)
    anchor_loc = np.expand_dims(anchor_loc, axis=0)
    ori_shape = np.expand_dims(np.asarray(s), axis=0)

    predictions = predict_func(image, anchor_cls, anchor_loc, ori_shape)

    loc_pred, cls_pred, loc_mask_label, loc_mask_pred, tot_loc_loss, nr_pos, pos_conf_loss, neg_conf_loss = predictions

    import pdb
    pdb.set_trace()

    boxes = postprocess(predictions, image_path=input_path, det_th=det_th)

    image_result = draw_result(ori_image, boxes)
    cv2.imwrite("debug_output.png", image_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')
    parser.add_argument('--det_th', help='detection threshold', type=float, default=0.25)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    predict_func = get_pred_func(args)

    predict_image(predict_func, float(args.det_th))
