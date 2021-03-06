import numpy as np

def ssd_anchor_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):

    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, 1), dtype=dtype)
    w = np.zeros((num_anchors, 1), dtype=dtype)

    for idx, s in enumerate(sizes):
        h[idx, 0] = s / img_shape[0]
        w[idx, 0] = s / img_shape[1]

    for idx, r in enumerate(ratios):
        h[idx + len(sizes), 0] = sizes[0] / img_shape[0] / np.sqrt(r)
        w[idx + len(sizes), 0] = sizes[0] / img_shape[1] * np.sqrt(r)

    y_reshape = np.reshape(y, (feat_shape[0] * feat_shape[1], 1))
    x_reshape = np.reshape(x, (feat_shape[0] * feat_shape[1], 1))

    y_reshape = np.repeat(y_reshape, len(sizes) + len(ratios), axis=0)
    x_reshape = np.repeat(x_reshape, len(sizes) + len(ratios), axis=0)

    w = np.tile(w, (feat_shape[0] * feat_shape[1], 1))
    h = np.tile(h, (feat_shape[0] * feat_shape[1], 1))


    xywh = np.concatenate([x_reshape, y_reshape, w, h], axis=1)

    xyxy = np.zeros_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # xyxy = np.minimum(np.maximum(xyxy, 0), 1)

    xywha = np.zeros((xyxy.shape[0], 6))

    xywha[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywha[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywha[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywha[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    xywha[:, 4] = xywha[:, 2] * xywha[:, 3] * 2
    xywha[:, 5] = xywha[:, 2] * xywha[:, 3] / 2

    return xywha

def ssd_anchor_all_layers(img_shape, layers_shape, anchor_sizes, anchor_ratios, anchor_steps, offset=0.5, dtype=np.float32):
    layer_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layer_anchors.append(anchor_bboxes)

    return np.concatenate(layer_anchors, axis=0)



