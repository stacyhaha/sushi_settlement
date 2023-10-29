# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name： MobileNet_classifier.py
   Author :
   time：
   last edit time:
   Description :   mobilenet classifier
-------------------------------------------------
"""
from PIL import Image


class MobileNetClassifier:
    def __init__(self):
        # Load model
        return

    def predict(self, image:Image, output_num=3):
        mock_result = [
            {"label": "Kohada", "prob": 0.92}, 
            {"label": "kos", "prob": 0.02},
            {"label": "Kohada", "prob": 0.002}
        ]
        return mock_result
        
