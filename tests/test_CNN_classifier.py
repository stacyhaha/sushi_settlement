# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_CNN_classifier.py
   Author :        
   time：          
   last edit time: 
   Description :   测试CNN分类模型
-------------------------------------------------
"""
import unittest
from PIL import Image
from sushi_settlement.classifier.CNN_classifier import CNNClassifier

class TestCNNClassifier(unittest.TestCase):
    def test_cnn_classifier(self):
        """
        测试方式：
        终端：
        python3 -m unittest tests/test_CNN_classifier.py
        """
        server = CNNClassifier()
        Input = "/Users/stacy/iss/5002project/backend/tests/images/Kohada.png"
        input_image = Image.open(Input)
        result = server.predict(input_image)
        import pprint
        pprint.pprint(result)
        