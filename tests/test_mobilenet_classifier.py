# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_mobilenet_classifier.py
   Author :        
   time：          
   last edit time: 
   Description :   测试CNN分类模型
-------------------------------------------------
"""
import unittest
from PIL import Image
from sushi_settlement.classifier.MobileNet_classifier import MobileNetClassifier

class TestMobileNectClassifier(unittest.TestCase):
    def test_mobilenet_classifier(self):
        """
        测试方式：
        终端：
        python3 -m unittest tests/test_mobilenet_classifier.py
        """
        server = MobileNetClassifier()
        Input = "/Users/stacy/iss/5002project/backend/tests/images/Kohada.png"
        input_image = Image.open(Input)
        result = server.predict(input_image)
        import pprint
        pprint.pprint(result)
        