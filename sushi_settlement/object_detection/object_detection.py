# # -*- coding utf-8 -*-
# """
# -------------------------------------------------
#    File Name：     object_detection.py
#    Author :
#    time：
#    last edit time:
#    Description :   目标检测
# -------------------------------------------------

import cv2
import time
import logging
import numpy as np
from PIL import Image, ImageDraw
import selectivesearch.selectivesearch as ss

logging.basicConfig(level=logging.INFO)
logger =logging.getLogger(__file__)

class ObjectDetection:
    def __init__(self):
        self.reduce_color = 51
        return
    
    def predict(self, image: Image):
        """
        detect object
        return the object location
        """
        image = image.resize((500, 400))
        img_array = np.array(image)
        img_array[:, :, 0] = img_array[:, :, 0] // self.reduce_color * self.reduce_color
        img_array[:, :, 1] = img_array[:, :, 1] // self.reduce_color * self.reduce_color
        img_array[:, :, 2] = img_array[:, :, 2] // self.reduce_color * self.reduce_color
        
        reduced_color= Image.fromarray(img_array[:,:,:3])
        reduced_color.show()
        reduced_color_array = np.array(reduced_color)

        bgr_image = cv2.cvtColor(reduced_color_array, cv2.COLOR_RGB2BGR)
       
        
        start = time.time()
        img_lbl, regions = ss.selective_search(bgr_image, scale=120, sigma=5, min_size=800)
        
        regions = list(set([i["rect"] for i in regions]))
        end = time.time()
        logger.info("[INFO] {} total region proposals".format(len(regions)))
        logger.info("[INFO] selective search took {:.4f} seconds".format(end - start))

        # post_process
        regions = [(i[0], i[1], i[0]+i[2], i[1]+i[3]) for i in regions]
        logger.info("[INFO] after filter, regions num is {}".format(len(regions)))
        return regions


if __name__ == "__main__":
    od = ObjectDetection()
    image = Image.open("/Users/stacy/iss/5002project/backend/tests/images/Shako_Nigiri.jpg")
    #image = Image.open("tests/images/sushi.png")
    image = image.resize((500, 400))
    regions = od.predict(image)

    draw = ImageDraw.Draw(image)
    for r in regions:
        draw.rectangle(r, outline="green", width=2)
    image.show()


    


