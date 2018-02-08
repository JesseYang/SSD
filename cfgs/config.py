from .config_coco_voc import cfg

cfg.lr_schedule = [(0, 1e-4), (6, 2e-4), (9, 2e-4), (12, 2e-4), (30, 2e-4), (80, 1e-4), (120, 1e-5)]
