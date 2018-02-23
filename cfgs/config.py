from .config_voc_0712 import cfg

# step based learning rate schedule
cfg.lr_schedule = [(0, 1e-3), (8e4, 1e-4), (1e5, 1e-5)]
