from .config_voc_2007 import cfg

# step based learning rate schedule
cfg.lr_schedule = [(0, 1e-3), (4e4, 1e-4), (48e3, 1e-5)]
