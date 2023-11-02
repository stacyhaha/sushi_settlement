# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     visualizer.py
   Author :        
   time：          
   last edit time: 
   Description :   根据location和模型结果，绘制识别框
-------------------------------------------------
"""
import os
import logging
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

class Visualizer:
    def __init__(self, font_path):
        self.font = ImageFont.truetype(font_path, size=17)
    
    def predict(self,image:Image, location):
        draw = ImageDraw.Draw(image)
        
        for r in location:
            draw.rectangle(r[:4], outline="green", width=2)
            draw.text((r[0], r[1]), "{}, prob: {:.2f}".format(r[-1], r[-2]), fill=(10, 10, 10), font=self.font)

        return image
    
