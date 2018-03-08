# from .config_voc_0712 import cfg
from .config_voc_0712_ssdlite import cfg

# step based learning rate schedule
cfg.lr_schedule = [(0, 1e-3), (4e4, 1e-4), (5e4, 1e-5)]
