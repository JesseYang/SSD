from .config_voc_0712 import cfg
# from .config_voc_0712_ssdlite import cfg
# from .config_coco_ssdlite import cfg
# from .config_voc_0712_ssdlite_1 import cfg

# step based learning rate schedule
# cfg.lr_schedule = [(0, 1e-3), (16e4, 1e-4), (20e4, 1e-5)]

# epoch based learning rate schedule
cfg.lr_schedule = [(0, 1e-3)]
