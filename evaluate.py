# coding: utf-8

import pdb
import numpy as np
import xml.etree.ElementTree as ET
import os
import shutil
import cv2
import pickle
import argparse
#from matplotlib import pyplot as plt
from os.path import basename

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]

        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    if not os.path.isfile(cachefile):
        # load annots

        # if cfg.gt_from_xml:
        if cfg.gt_format == "voc":
            with open(cfg.imagesetfile, 'r') as f:
                lines = f.readlines()
            imagenames = [x.strip() for x in lines]
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = parse_rec(cfg.annopath.format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))

        else:
            recs = {}
            npos = 0
            with open(cfg.test_list) as f:
                lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            for idx, image_id in enumerate(image_ids):
                if idx % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        idx + 1, len(image_ids)))
                objects = []
                record = splitlines[idx]
                i = 1
                while i < len(record):
                    # for each ground truth box
                    xmin = int(np.round(float(record[i])))
                    ymin = int(np.round(float(record[i + 1])))
                    xmax = int(np.round(float(record[i + 2])))
                    ymax = int(np.round(float(record[i + 3])))
                    class_idx = int(record[i + 4])
                    i += 5

                    obj_struct = {}
                    obj_struct['name'] = cfg.classes_name[class_idx]
                    obj_struct['difficult'] = 0
                    obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
                    objects.append(obj_struct)
                recs[image_id] = objects

        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in recs.keys():
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}




    # read dets
    detfile = detpath.format(classname)
    if os.path.isfile(detfile):
        with open(detfile, 'r') as f:
            lines = f.readlines()
    else:
        lines = []

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def do_python_eval(res_prefix, verbose=True):
    _devkit_path = 'VOCdevkit'
    _year = '2007'
    
    #filename = '/data/hongji/darknet/results/comp4_det_test_{:s}.txt' 
    filename = res_prefix + '{:s}.txt'
    cachedir = 'annotations_cache'
    if os.path.isdir(cachedir):
        shutil.rmtree(cachedir)
    aps = []
    use_07_metric = True

    for i, cls in enumerate(cfg.classes_name):
        
        rec, prec, ap = voc_eval(
            filename, cls, cachedir, ovthresh=cfg.iou_th,
            use_07_metric=use_07_metric)
        aps += [ap]
        if verbose:
            print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    if verbose:
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    return np.mean(aps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='path of the test file', default='voc_2007_test.txt')
    args = parser.parse_args()

    do_python_eval("result_pred/")
