# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_assembler.py
   Author :        
   time：          
   last edit time: 
   Description :   测试融合模型
-------------------------------------------------
"""
import unittest
from PIL import Image
from sushi_settlement.assembler.assembler import Assembler

class TestAssembler(unittest.TestCase):
    def test_assembler(self):
        """
        测试方式：
        终端：
        python3 -m unittest tests/test_assembler.py
        """
        server = Assembler()
        mock_input = {
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
        result = server.predict(mock_input)
        import pprint
        pprint.pprint(result)
        