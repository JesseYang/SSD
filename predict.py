#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2
import re

# %matplotlib inline
import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import skimage.io as io
# import pylab

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

try:
    from .cfgs.config import cfg
    from .utils_bak import postprocess
except Exception:
    from cfgs.config import cfg
    from utils_bak import postprocess

try:
    from .vgg_ssd import VGGSSD
    from .vgg_fssd import VGGFSSD
    from .mobilenetv2_ssd import SSDLite
except Exception:
    from vgg_ssd import VGGSSD
    from vgg_fssd import VGGFSSD
    from mobilenetv2_ssd import SSDLite

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    if args.network == "vgg_fssd":
        model = VGGFSSD()
    elif args.network == "vgg_ssd":
        model = VGGSSD()
    else:
        model = SSDLite()
    # model = VGGSSD() if args.backbone == 'vgg' else SSDLite(data_format="NCHW")
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["loc_pred", "cls_pred"])

    predict_func = OfflinePredictor(predict_config)
    return predict_func

def draw_result(image, boxes):
    colors = [(255,0,0), (0,255,0), (0,0,255),
              (255,255,0), (255,0,255), (0,255,255),
              (122,0,0), (0,122,0), (0,0,122),
              (122,122,0), (122,0,122), (0,122,122)]

    text_colors = [(0,255,255), (255,0,255), (255,255,0),
                  (0,0,255), (0,255,0), (255,0,0),
                  (0,122,122), (122,0,122), (122,122,0),
                  (0,0,122), (0,122,0), (122,0,0)]

    image_result = np.copy(image)
    k_idx = 0
    for klass, k_boxes in boxes.items():
        for k_box in k_boxes:

            [conf, xmin, ymin, xmax, ymax] = k_box

            label = "%s %.3f" % (klass, conf)
            label_height = 16
            label_width = len(label) * 10
 
            cv2.rectangle(image_result,
                          (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)),
                          colors[k_idx % len(colors)],
                          3)
            cv2.rectangle(image_result,
                          (int(xmin) - 2, int(ymin) - label_height),
                          (int(xmin) + label_width, int(ymin)),
                          colors[k_idx % len(colors)],
                          -1)
            cv2.putText(image_result,
                        label,
                        (int(xmin), int(ymin) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_colors[k_idx % len(text_colors)])
        k_idx += 1

    return image_result

def predict_image(input_path, output_path, predict_func, det_th):
    ori_image = cv2.imread(input_path)
    cvt_clr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(cvt_clr_image, (cfg.img_w, cfg.img_h))
    image = np.expand_dims(image, axis=0)
    predictions = predict_func(image)

    import pdb
    pdb.set_trace()

    boxes = postprocess(predictions, image_path=input_path, det_th=det_th)

    image_result = draw_result(ori_image, boxes)
    cv2.imwrite(output_path, image_result)

def generate_pred_result(image_paths, predict_func, pred_dir):

    for class_name in cfg.classes_name:
        with open(os.path.join(pred_dir, class_name + ".txt"), 'w') as f:
            continue

    for image_idx, image_path in enumerate(image_paths):
        if image_idx % 100 == 0 and image_idx > 0:
            print(str(image_idx))
        
        image_id = os.path.basename(image_path).split('.')[0]

        ori_image = cv2.imread(image_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(ori_image, (cfg.img_w, cfg.img_h))
        image = np.expand_dims(image, axis=0)
        predictions = predict_func(image)

        pred_results = postprocess(predictions, image_path=image_path)

        for class_name in pred_results.keys():
            with open(os.path.join(pred_dir, class_name + ".txt"), 'a') as f:
                for box in pred_results[class_name]:
                    record = [image_id]
                    record.extend(box)
                    record = [str(ele) for ele in record]
                    f.write(' '.join(record) + '\n')

def generate_pred_images(image_paths, predict_func, crop, generate_resultfomat, evaluate_name, output_dir, det_th, enlarge_ratio=1.3):
    json_file = []
   
    for image_idx, image_path in enumerate(image_paths):
        # if image_idx >= 2:
        #     continue

        if not os.path.exists(image_path):
            continue
        if image_idx % 100 == 0 and image_idx > 0:
            print(str(image_idx))
        # print(image_path)
        img_name = image_path.split('/')[-1].split('.')[0]
        ori_image = cv2.imread(image_path)

        cvt_color_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(cvt_color_image, (cfg.img_w, cfg.img_h))
        image = np.expand_dims(image, axis=0)
        predictions = predict_func(image)

        boxes = postprocess(predictions, image_path=image_path, det_th=det_th)

        image_name = ntpath.basename(image_path)
        if crop == True:
            # crop each box and save
            for klass, k_boxes in boxes.items():
                for box_idx, k_box in enumerate(k_boxes):
                    [conf, xmin, ymin, xmax, ymax] = k_box
                    xcenter = (xmin + xmax) / 2
                    ycenter = (ymin + ymax) / 2
                    width = (xmax - xmin) * enlarge_ratio
                    height = (ymax - ymin) * enlarge_ratio
                    xmin = np.max([0, int(xcenter - width / 2)])
                    ymin = np.max([0, int(ycenter - height / 2)])
                    xmax = np.min([ori_image.shape[1] - 1, int(xcenter + width / 2)])
                    ymax = np.min([ori_image.shape[0] - 1, int(ycenter + height / 2)])
                    crop_img = ori_image[int(ymin):int(ymax),int(xmin):int(xmax)]

                    name_part, img_type = image_name.split('.')
                    save_name = name_part + "_" + klass + "_" + str(box_idx) + "." + img_type
                    save_path = os.path.join(output_dir, save_name)
                    if generate_resultfomat:
                        result_dict = {}
                        result_dict["image_id"] = int(re.sub(r"\b0*([1-9][0-9]*|0)", r"\1", img_name))
                        result_dict["category_id"] = int(cfg.classes_label[klass])
                        result_dict["bbox"] = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
                        result_dict["score"] = float(round(conf, 3))
                        json_file.append(json.dumps(result_dict))

                    else:
                        cv2.imwrite(save_path, crop_img)
                    
        else:
            # draw box on original image and save
            image_result = draw_result(ori_image, boxes)
            # save_path = os.path.join(output_dir, str(uuid.uuid4()) + ".jpg")
            save_path = os.path.join(output_dir, image_path.split('/')[-1])
            # cv2.imwrite(save_path, image_result)
            cv2.imwrite(save_path, image_result)

    if generate_resultfomat:
        save_json = open(evaluate_name, 'w')
        save_json.write("[")
        for idx, i in enumerate(json_file):
            
            save_json.write(i)

            if idx == (len(json_file)-1):
                save_json.write("]")
            else:
                save_json.write(",")
        save_json.close()
        

def evaluate_map(annotations, evaluate_name):
    pass

    '''
    print("strat evaluate......")
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    # print (annType)
    
    #initialize COCO ground truth api
    # dataDir="/home/user/Datasets/coco"
    # dataType='val2017'
    # annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    # annFile = annotations
    # print(annFile)
    print(annotations)
    cocoGt=COCO(annotations)

    #initialize COCO detections api
    # resFile='%s/%s_%s_fake%s100_results.json'
    # resFile = resFile%(dataDir, prefix, dataType, annType)
    resFile = os.path.join('/home/user/yzx/SSD',evaluate_name)
    
    if not os.path.exists(resFile):
        print(resFile,"not exists")
        quit()
    print(resFile)
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    '''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='the network type', default='vgg_fssd')
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='output.png')
    parser.add_argument('--test_path', help='path of the test file', default=None)
    parser.add_argument('--pred_dir', help='directory to save txt result', default='result_pred')
    parser.add_argument('--det_th', help='detection threshold', type=float, default=0.15)
    parser.add_argument('--gen_image', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--output_dir', help='directory to save image result', default='output')

    ##evaluate parameter
    parser.add_argument('--generate_resultfomat', help='if generate json file for coco map compute', action='store_true')
    parser.add_argument('--evaluate_name', help='the evaluate result .json file name', default='train2017_result.json')
    parser.add_argument('--annotations_path', help='path of annotations json file', default='coco/annotations/instances_train2017.json')
    parser.add_argument('--map_evaluate', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if args.map_evaluate:
        evaluate_map(args.annotations_path, args.evaluate_name)
        quit()

    predict_func = get_pred_func(args)
    
    new_dir = args.output_dir if args.gen_image else args.pred_dir

    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    if args.input_path != None:
        # predict one image (given the input image path) and save the result image
        predict_image(args.input_path, args.output_path, predict_func, args.det_th)
    elif args.test_path != None:
        test_paths = args.test_path.split(',')
        image_paths = []
        for test_path in test_paths:
            with open(test_path) as f:
                content = f.readlines()
            image_paths.extend([line.split(' ')[0].strip() for line in content])
                
        print("Number of images to predict: " + str(len(image_paths)))
        if args.gen_image:
            # given the txt file, predict the images and save the images result
            generate_pred_images(image_paths, predict_func, args.crop, args.generate_resultfomat, args.evaluate_name, args.output_dir, float(args.det_th))
        else:
            # given the txt file, predict the images and save the txt result
            generate_pred_result(image_paths, predict_func, args.pred_dir)

# note:
#     if only predict image from .txt  file:
#         python3 predict.py --model_path train_log/ssdlite_coco/model-238160 --data_format NCHW --test_path coco_val.txt --gen_image --crop

    
#     if want to evaluate model,first genterate .json file
#         python3 predict.py --model_path train_log/ssdlite_coco/model-238160 --data_format NCHW --test_path coco_val.txt --gen_image --crop --generate_resultfomat

          
          # then python3 predict.py --map_evaluate
