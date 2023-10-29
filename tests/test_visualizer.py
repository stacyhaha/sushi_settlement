# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_visualizer.py
   Author :        
   time：          
   last edit time: 
   Description :   测试pdf生成模块
-------------------------------------------------
"""
import os
import json
import unittest
from PIL import Image
from sushi_settlement.visualizer.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def test_visualizer(self):
        """
        测试方式：
        终端：
        python3 -m unittest tests/test_visualizer.py
        """
        server = Visualizer()
        Input = "/Users/stacy/iss/5002project/backend/tests/images/Kohada.png"
        input_image = Image.open(Input)
        location = (20, 20, 34, 34)
        model_res = [
            {"label": "Kohada", "prob": 0.92}, 
            {"label": "kos", "prob": 0.02},
            {"label": "Kohada", "prob": 0.002}
        ]
        
        result = server.predict(input_image, location, model_res)
        import pprint
        pprint.pprint(result)
        