from easydict import EasyDict as edict
import numpy as np
from .config_utils import *
# import itertools.product as product

cfg = edict()

# cfg.lr_schedule = [(0, 1e-3), (80000, 1e-4), (100000, 1e-5)]
cfg.lr_schedule = [(0, 1e-3), (2068*150, 1e-4), (2068*200, 1e-5), (2068*250, 1e-6)]

cfg.initial_lr = 1e-3
cfg.warm_epoch = 1 # max epoch for retraining
cfg.gamma = 0.1 # Gamma update for momentum SGD

cfg.max_itr = 120000    # 120k
cfg.max_epoch = 300

cfg.img_size = 300
cfg.img_w = cfg.img_size
cfg.img_h = cfg.img_size

cfg.n_boxes = 5

cfg.threshold = 0.5

cfg.weight_decay = 5e-4

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.class_num = len(cfg.classes_name)

cfg.feat_shapes = [38, 19, 10, 5, 3, 1]

cfg.anchor_sizes = [[0.1, np.sqrt(0.1 * 0.2)],
                    [0.2, np.sqrt(0.2 * 0.37)],
                    [0.37, np.sqrt(0.37 * 0.54)],
                    [0.54, np.sqrt(0.54 * 0.71)],
                    [0.71, np.sqrt(0.71 * 0.88)],
                    [0.88, np.sqrt(0.88 * 1.05)]]
cfg.anchor_sizes = np.asarray(cfg.anchor_sizes) * cfg.img_size

cfg.anchor_ratios = [[2, 3],
                     [2, 3],
                     [2, 3],
                     [2, 3],
                     [2],
                     [2]]
cfg.anchor_ratios = np.asarray(cfg.anchor_ratios)

cfg.anchor_steps=[8, 16, 32, 64, 100, 300]

cfg.all_anchors = ssd_anchor_all_layers(cfg.img_size,
                                        cfg.feat_shapes,
                                        cfg.anchor_sizes,
                                        cfg.anchor_ratios,
                                        cfg.anchor_steps)



'''parameters for all_anchors calculated by itertools method'''

# cfg.feature_maps = [38, 19, 10, 5, 3, 1]
# cfg.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# mean = []
# for k, f in enumerate(cfg.feature_maps):
#     for i, j in product(range(f), repeat=2):
#         f_k = cfg.img_size / cfg.anchor_steps[k]
#         cx = (j + 0.5) / f_k
#         cy = (i + 0.5) / f_k
#         s_k = cfg.anchor_sizes[k, 0] / cfg.img_size
#         s_k_prime = cfg.anchor_sizes[k, 1] / cfg.img_size
#         mean += [cx, cy, s_k, s_k]
#         mean += [cx, cy, s_k_prime, s_k_prime]
#         for ar in cfg.aspect_ratios:
#             mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
#             mean += [cx, cy, s_k /sqrt(ar), s_k * sqrt(ar)]
# cfg.all_anchors_ = np.array(mean).reshape(-1, 4)
# np.clip(cfg.all_anchors_, 0, 1, out=cfg.all_anchors_)

'''end'''

cfg.tot_anchor_num = cfg.all_anchors.shape[0]

cfg.prior_scaling = [0.1, 0.2]

cfg.random_crop = True
cfg.random_expand = True
cfg.random_distort = True
cfg.random_inter = True

cfg.classes_num = { }
for idx, name in enumerate(cfg.classes_name):
    cfg.classes_num[name] = idx

cfg.train_list = ["voc_2007_train.txt", "voc_2012_train.txt", "voc_2007_val.txt", "voc_2012_val.txt"]
cfg.test_list = "voc_2007_test_without_diff.txt"

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

cfg.mAP = True

cfg.neg_ratio = 3

cfg.gt_from_xml = True
cfg.gt_format = "voc"
cfg.annopath = 'voc/VOCdevkit/VOC2007/Annotations/{:s}.xml'
cfg.imagesetfile = 'voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
