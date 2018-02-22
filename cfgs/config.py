from .config_voc_0712 import cfg

# step based learning rate schedule
cfg.lr_schedule = [(0, 1e-3), (6e4, 1e-4), (7e4, 1e-5)]
