import numpy as np
from math import sqrt as sqrt
from itertools import product as product

def ssd_anchor_all_layers(img_size, feat_shapes, anchor_sizes, anchor_ratios, anchor_steps, offset=0.5, dtype=np.float32):
    mean = []
    for k, shape in enumerate(feat_shapes):
        for i, j in product(range(shape), repeat=2):
            f_k = img_size / anchor_steps[k]
            cx = (i + 0.5) / f_k
            cy = (j + 0.5) / f_k

            s_k = anchor_sizes[k][0] / img_size
            mean += [cx, cy, s_k, s_k]

            s_k_prime = anchor_sizes[k][1] / img_size
            mean += [cx, cy, s_k_prime, s_k_prime]

            for ar in anchor_ratios[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

    anchor_num = len(mean) // 4
    output = np.asarray(mean).reshape((anchor_num, 4))
    output = np.clip(output, 0, 1)

    return output
