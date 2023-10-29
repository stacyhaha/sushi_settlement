# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     assembler.py
   Author :
   time：
   last edit time:
   Description :   融合模型
-------------------------------------------------
"""
from PIL import Image


class Assembler:
    def __init__(self):
        return

    def predict(self, model_res):
        #mock
        model_res = {
            "cnn": [
                {"label": "Kohada", "prob": 0.92}, 
                {"label": "kos", "prob": 0.02},
                {"label": "Kohada", "prob": 0.002}
                ],
            "mobilenet": [
                {"label": "Kohada", "prob": 0.92}, 
                {"label": "kos", "prob": 0.02},
                {"label": "Kohada", "prob": 0.002}
            ]
        }

        mock_result = [
                {"label": "Kohada", "prob": 0.92}, 
                {"label": "kos", "prob": 0.02},
                {"label": "Kohada", "prob": 0.002}
                ]
        return mock_result
        
