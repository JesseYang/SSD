from easydict import EasyDict as edict
import numpy as np
from .config_utils import *

cfg = edict()

cfg.img_size = 300
cfg.img_w = cfg.img_size
cfg.img_h = cfg.img_size

cfg.n_boxes = 5

cfg.threshold = 0.6

cfg.weight_decay = 5e-4

cfg.leaky_k = 0.125

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.class_num = len(cfg.classes_name)

cfg.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]

cfg.anchor_sizes = [[0.1, np.sqrt(0.1 * 0.34)],
                    [0.34, np.sqrt(0.34 * 0.48)],
                    [0.48, np.sqrt(0.48 * 0.62)],
                    [0.62, np.sqrt(0.62 * 0.76)],
                    [0.76, np.sqrt(0.76 * 0.9)],
                    [0.9, np.sqrt(0.9 * 1.04)]]
cfg.anchor_sizes = np.asarray(cfg.anchor_sizes) * cfg.img_size

cfg.anchor_ratios = [[2, 0.5],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5, 3, 1/3],
                     [2, 0.5],
                     [2, 0.5]]
cfg.anchor_ratios = np.asarray(cfg.anchor_ratios)

cfg.anchor_steps=[8, 16, 32, 64, 100, 300]

cfg.all_anchors = ssd_anchor_all_layers([cfg.img_size, cfg.img_size],
                                        cfg.feat_shapes,
                                        cfg.anchor_sizes,
                                        cfg.anchor_ratios,
                                        cfg.anchor_steps)

cfg.tot_anchor_num = cfg.all_anchors.shape[0]

cfg.prior_scaling = [0.1, 0.1, 0.2, 0.2]
# cfg.prior_scaling = [1.0, 1.0, 1.0, 1.0]

cfg.classes_num = { }
for idx, name in enumerate(cfg.classes_name):
    cfg.classes_num[name] = idx

cfg.train_list = ["voc_2007_train.txt", "voc_2007_val.txt"]
cfg.test_list = "voc_2007_test_without_diff.txt"

cfg.train_sample_num = 0
for train_file in cfg.train_list:
    f = open(train_file, 'r')
    cfg.train_sample_num += len(f.readlines())

cfg.det_th = 0.01
cfg.iou_th = 0.5
cfg.neg_iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.hard_sample_mining = True

cfg.alpha = 1.0
cfg.freeze_backbone = False

cfg.mAP = True

cfg.neg_ratio = 3

cfg.gt_from_xml = True
cfg.gt_format = "voc"
cfg.annopath = 'voc/VOCdevkit/VOC2007/Annotations/{:s}.xml'
cfg.imagesetfile = 'voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
