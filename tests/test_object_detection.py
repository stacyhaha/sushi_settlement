# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_object_detection
   Author :        
   time：          
   last edit time: 
   Description :   测试目标检测
-------------------------------------------------
"""

import os
import json
import unittest
from PIL import Image
from sushi_settlement.object_detection.object_detection import ObjectDetection

class TestObjectDetection(unittest.TestCase):
    def test_object_detection(self):
        """
        测试方式：
        终端：
        python3 -m unittest tests/test_object_detection.py
        """
        server = ObjectDetection()
        Input = "/Users/stacy/iss/5002project/backend/tests/images/Kohada.png"
        input_image = Image.open(Input)
        result = server.predict(input_image)
        import pprint
        pprint.pprint(result)