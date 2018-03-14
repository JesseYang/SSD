from easydict import EasyDict as edict
import numpy as np
from .config_utils import *

cfg = edict()

cfg.img_size = 320
cfg.img_w = cfg.img_size
cfg.img_h = cfg.img_size

cfg.n_boxes = 5

cfg.threshold = 0.6

cfg.weight_decay = 5e-4

cfg.leaky_k = 0.125

cfg.classes_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cfg.class_num = len(cfg.classes_name)

cfg.feat_shapes = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]

cfg.anchor_sizes = [[0.07, np.sqrt(0.07 * 0.15)],
                    [0.15, np.sqrt(0.15 * 0.37)],
                    [0.37, np.sqrt(0.37 * 0.54)],
                    [0.54, np.sqrt(0.54 * 0.71)],
                    [0.71, np.sqrt(0.71 * 0.88)],
                    [0.88, np.sqrt(0.88 * 1.05)]]
cfg.anchor_sizes = np.asarray(cfg.anchor_sizes) * cfg.img_size

cfg.anchor_ratios = [[2, 0.5],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5],
                     [2, 0.5]]
cfg.anchor_ratios = np.asarray(cfg.anchor_ratios)

cfg.anchor_steps=[16, 32, 64, 106, 160, 320]

cfg.all_anchors = ssd_anchor_all_layers([cfg.img_size, cfg.img_size],
                                        cfg.feat_shapes,
                                        cfg.anchor_sizes,
                                        cfg.anchor_ratios,
                                        cfg.anchor_steps)

cfg.tot_anchor_num = cfg.all_anchors.shape[0]

cfg.prior_scaling = [0.1, 0.1, 0.2, 0.2]

cfg.random_crop = True
cfg.random_expand = True
cfg.random_inter = True

cfg.classes_num = { }
for idx, name in enumerate(cfg.classes_name):
    cfg.classes_num[name] = idx

cfg.train_list = ["coco_train.txt"]
cfg.test_list = "coco_val.txt"

cfg.train_sample_num = 0
for train_file in cfg.train_list:
    f = open(train_file, 'r')
    cfg.train_sample_num += len(f.readlines())

cfg.max_gt_box_shown = 30
cfg.det_th = 0.01
cfg.iou_th = 0.5
cfg.neg_iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.hard_sample_mining = True

cfg.alpha = 1.0
cfg.freeze_backbone = False

cfg.mAP = False

cfg.neg_ratio = 3

cfg.gt_from_xml = True
cfg.gt_format = "voc"
cfg.annopath = 'voc/VOCdevkit/VOC2007/Annotations/{:s}.xml'
cfg.imagesetfile = 'voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
