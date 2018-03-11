# from .config_voc_0712 import cfg
from .config_voc_0712_ssdlite import cfg

# step based learning rate schedule
cfg.lr_schedule = [(0, 1e-3), (8e4, 1e-4), (10e4, 1e-5)]
