from flask import Flask, render_template, request, Response

import torch
import torchvision

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
import cv2
import json

from pattern.text.en import singularize


# Initialize Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
predictor = DefaultPredictor(cfg)

# Initialize Flask
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def find_contour(mask: torch.Tensor) -> list:
    arr = []

    for i, row in enumerate(mask):
        occ = np.where(mask[i])[0]
        
        if len(occ) == 1:
            arr.append((occ[0], i))
        elif len(occ) >= 2:
            arr.append((occ[0], i))
            arr.append((occ[-1], i))

    return arr


@app.route('/api/image', methods=['POST'])
def image():
    label = singularize(request.form['label'])
    shape_x = int(request.form['shape_x'])
    shape_y = int(request.form['shape_y'])
    image = request.files['image']

    label_id = classes.index(label)

    image = image.read()
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    height, width = image.shape[:2]

    outputs = predictor(image)

    pred_classes = outputs['instances'].pred_classes
    pred_boxes = outputs['instances'].pred_boxes
    pred_masks = outputs['instances'].pred_masks

    indices = np.where(pred_classes == label_id)[0]

    grid = set()

    for index in indices:
        countour = find_contour(pred_masks[index])

        for pos in countour:
            for col in range(shape_x):
                if pos[0] < width / shape_x * (col + 1):
                    x = col
                    break

            for row in range(shape_y):
                if pos[1] < height / shape_y * (row + 1):
                    y = row
                    break

            grid.add((x, y))

    return json.dumps(list(grid))