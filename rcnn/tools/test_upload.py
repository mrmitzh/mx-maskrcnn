# -*- coding: utf-8 -*-
import argparse
from ..config import default, generate_config
from ..symbol import *
from ..utils.load_model import load_param
from ..core.module import MutableModule
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
bbox_pred = nonlinear_pred

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess

def str2bool(v):
    if v.lower() in ('True', 'true'):
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)    
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', type=str2bool)
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--image_name', help='image file path',type=str)
    
    
    args = parser.parse_args()

    return args


args = parse_args()
ctx = mx.gpu(args.gpu)


def demo_maskrcnn(network, ctx, prefix, epoch,img_path,
                   vis= True, has_rpn = True, thresh = 0.001):
    
    assert has_rpn,"Only has_rpn==True has been supported."
    sym = eval('get_' + network + '_mask_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
    
    max_image_shape = (1,3,4096,4096)
    #max_image_shape = (1,3,1024,1024)
    max_data_shapes = [("data",max_image_shape),("im_info",(1,3))]
    mod = MutableModule(symbol = sym, data_names = ["data","im_info"], label_names= None,
                            max_data_shapes = max_data_shapes,
                              context=ctx)
    mod.bind(data_shapes = max_data_shapes, label_shapes = None, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    class OneDataBatch():
        def __init__(self,img):
            im_info = mx.nd.array([[img.shape[0],img.shape[1],1.0]])
            img = np.transpose(img,(2,0,1)) 
            img = img[np.newaxis,(2,1,0)]
            self.data = [mx.nd.array(img),im_info]
            self.label = None
            self.provide_label = None
            self.provide_data = [("data",(1,3,img.shape[2],img.shape[3])),("im_info",(1,3))]
    
    img_ori = cv2.imread(img_path)
    img_oriX = img_ori.shape[1]
    img_oriY = img_ori.shape[0]
    
    batch = OneDataBatch(img_ori)
    mod.forward(batch, False)
    results = mod.get_outputs()
    output = dict(zip(mod.output_names, results))
    rois = output['rois_output'].asnumpy()[:, 1:]


    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
    mask_output = output['mask_prob_output'].asnumpy()

    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, [img_ori.shape[0],img_ori.shape[1]])

    nms = py_nms_wrapper(config.TEST.NMS)

    boxes= pred_boxes

    CLASSES  = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'mcycle', 'bicycle')

    all_boxes = [[[] for _ in xrange(1)]
                 for _ in xrange(len(CLASSES))]
    all_masks = [[[] for _ in xrange(1)]
                 for _ in xrange(len(CLASSES))]
    label = np.argmax(scores, axis=1)
    label = label[:, np.newaxis]

    for cls in CLASSES:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_masks = mask_output[:, cls_ind, :, :]
        cls_scores = scores[:, cls_ind, np.newaxis]
        #print cls_scores.shape, label.shape
        keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
        cls_masks = cls_masks[keep, :, :]
        dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
        keep = nms(dets)
        #print dets.shape, cls_masks.shape
        all_boxes[cls_ind] = dets[keep, :]
        all_masks[cls_ind] = cls_masks[keep, :, :]

    boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]
    masks_this_image = [[]] + [all_masks[j] for j in range(1, len(CLASSES))]


    import copy
    import random
    import json
    class_names = CLASSES
    color_white = (255, 255, 255)
    scale = 1.0
    im = copy.copy(img_ori)
    json_dict = {}
    detect_counter = 0

    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        masks = masks_this_image[j]
        for i in range(len(dets)):
            bbox = dets[i, :4] * scale
            if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]  :
                continue
            score = dets[i, -1]
            bbox = map(int, bbox)
            tmp = {}
            tmp["x1"] = bbox[0]
            tmp["y1"] = bbox[1]
            tmp["x2"] = bbox[2]
            tmp["y2"] = bbox[3]
            tmp["class_name"] = class_names[j]
            tmp["scores"] = int(100*score)
            json_dict[detect_counter] = tmp
            detect_counter += 1
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            mask = masks[i, :, :]
            mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask_color = random.randint(0, 255)
            c = random.randint(0, 2)
            target = im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] + mask_color * mask
            target[target >= 255] = 255
            im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] = target
    im = im[:,:,(2,1,0)]
    plt.imshow(im)
    json_ret = json.dumps(json_dict)
    
#    plt.savefig('figures/test_result.jpg')
    if vis:
        plt.show()
    else:
        plt.savefig('figures/test_result.jpg')
    return json_ret







ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            
            detect_json = demo_maskrcnn(network = args.network, 
                  ctx = ctx,
                  prefix = args.prefix,
                  epoch = args.epoch, 
                  img_path = filename,
                  vis= args.vis, 
                  has_rpn = True,
                  thresh = args.thresh)
            print detect_json
            return html + '<br><img src=' + file_url + '>' + '<h2>'+ detect_json +'</h2>'
    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0')
