# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     settlement.py
   Author :        
   time：          
   last edit time: 
   Description :   integrate all parts
-------------------------------------------------
"""
import os
import logging
from PIL import Image
from .object_detection.object_detection import ObjectDetection
from .classifier.CNN_classifier import CNNClassifier
from .classifier.MobileNet_classifier import MobileNetClassifier
from .assembler.assembler import Assembler
from .visualizer.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

class Settlement:
    def __init__(self, workspace, CNN_model_path, MobileNet_model_path, font_path):
        self.workspace = workspace
        if not os.path.exists(workspace):
            os.makedirs(workspace)

        self.object_detection = ObjectDetection()
        self.CNN_classifier = CNNClassifier(CNN_model_path)
        self.MobileNet_classifier = MobileNetClassifier(MobileNet_model_path)
        self.assembler = Assembler()
        self.visualizer = Visualizer(font_path)
        logger.info("initiate succeessfully")

    def predict(self, image_path):
        image = Image.open(image_path)
        locations = self.object_detection.predict(image)

        # todo multi process
        res = {}
        for sub_loc in locations:
            assemble_res = self.model_predict(image, sub_loc)
            res[tuple(sub_loc)] = assemble_res
        
        res = self.post_process_model_res(res)
        nms_res = self.nms(res)
        logger.info("[INOF] after nms the region num is {}".format(len(nms_res)))

        image_detected = self.visualizer.predict(image, nms_res)
        image_detected_path = image_path.split(".")[0] + "_detected.jpg"
        image_detected.save(image_detected_path)
        logger.info(f"[INFO] save detect image in {image_detected_path}")

    
    def postprocess_res(self, res):
        return



    def model_predict(self, image:Image, sub_loc):
        model_res = {}
        croped_img = image.crop(sub_loc)
        cnn_res = self.CNN_classifier.predict(croped_img)
        mobilenet_res = self.MobileNet_classifier.predict(croped_img)
        model_res["cnn"] = cnn_res
        model_res["mobilenet"] = mobilenet_res
        assemble_res = self.assembler.predict(model_res)
        return assemble_res
    
    def post_process_model_res(self, asssemble_res):
        """
        only retain the highest probability label of every region
        """
        res = {}
        for sub_loc, v in asssemble_res.items():
            res[sub_loc] = v[0]
        return res

    def IOU(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2- y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter*1.0 / area_union


        # determine if box1 contains box2
        if (x3 <= x1 <= x4 and   x3 <= x2 <= x4 and  y3 <= y1 <= y4 and y3 <= y2 <= y4) or \
        (x1 <= x3 <= x2 and   x1 <= x4 <= x2 and  y1 <= y3 <= y2 and y1 <= y4 <= y2):
            return max(iou, 0.5)
        return iou
    
    def nms(self, boxes, conf_threshold=0.3, iou_threshold=0.5):
        boxes = [(*key, value["prob"], value["label"]) for key, value in boxes.items()]
        bbox_list = []
        
        boxes = list(filter(lambda x: x[4] >= conf_threshold, boxes))
        boxes_sorted = sorted(boxes, reverse=True, key=lambda x:x[4])
        
        while len(boxes_sorted) > 0:
            current_box = boxes_sorted.pop(0)
            bbox_list.append(current_box)
            for box in boxes_sorted[::-1]:
                if current_box[5] == box[5]: # the same label
                    iou = self.IOU(current_box[:4], box[:4])
                    if iou >= iou_threshold:
                        boxes_sorted.remove(box)
        return bbox_list
        

if __name__ == "__main__":
    settlement = Settlement(
        workspace = "/Users/stacy/iss/workspace2",
        CNN_model_path = r'/Users/stacy/iss/5002project/backend/sushi_settlement/models/cnn_sushi.h5',
        MobileNet_model_path = r'/Users/stacy/iss/5002project/backend/sushi_settlement/models/mobilenet_sushi.h5',
        font_path="/Users/stacy/iss/5002project/backend/UNSII-2.ttf"
    )
    image_path = "/Users/stacy/iss/5002project/backend/tests/images/sushi.png"
    settlement.predict(image_path)
