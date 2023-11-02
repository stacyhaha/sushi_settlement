# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     CNN_classifier.py
   Author :
   time：
   last edit time:
   Description :   CNN model
-------------------------------------------------
"""
from PIL import Image
import tensorflow as tf
import numpy as np

class_label_to_name = {
    '1': 'Abokado_Maki',
    '2': 'Aji',
    '3': 'Akagai',
    '4': 'Akami',
    '5': 'Amaebi',
    '6': 'Ankimo',
    '7': 'Aoyagi',
    '8': 'Awabi',
    '9': 'Battera_Sushi',
    '10': 'California_Roll',
    '11': 'Daikon_Oshinko_Maki',
    '12': 'Ebi_Nigiri',
    '13': 'Funazushi',
    '14': 'Futomaki',
    '15': 'Gyu_Nigiri',
    '16': 'Hamaguri',
    '17': 'Hamo',
    '18': 'Hirame_&_Karei',
    '19': 'Hokkigai',
    '20': 'Hotate_Nigiri',
    '21': 'Ika_Nigiri',
    '22': 'Ikura_Gunkan',
    '23': 'Inarizushi',
    '24': 'Kamaboko_Kani',
    '25': 'Kanpyomaki',
    '26': 'Kappa_Maki',
    '27': 'Kazunoko',
    '28': 'Kohada',
    '29': 'Kuruma',
    '30': 'Meharizushi',
    '31': 'Mirugai',
    '32': 'Natto_Maki',
    '33': 'Negitoro',
    '34': 'Saba',
    '35': 'Sake_Nigiri',
    '36': 'Sayori',
    '37': 'Shako_Nigiri',
    '38': 'Shirasu',
    '39': 'Shiroebi',
    '40': 'Tai_&_Madai',    
    '41': 'Tako_Nigiri',
    '42': 'Tamagoyaki',
    '43': 'Tekkamaki',
    '44': 'Temaki',
    '45': 'Temarizushi',
    '46': 'Tobiko_Nigiri',
    '47': 'Torigai',
    '48': 'Toro',
    '49': 'Unagi',
    '50': 'Uni'
}

class CNNClassifier:
    def __init__(self, model_path):
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        return

    def predict(self, img, top_n=3):
        # 将图像转换为3通道的彩色图像
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((224, 224))
        img = np.asarray(img)
        img = img / 255.0  # 归一化像素值
        img = np.expand_dims(img, axis=0)  # 添加批处理维度

        # 使用模型进行预测
        predictions = self.model.predict(img)

        # 获取前top_n的类别和概率
        top_indices = np.argpartition(predictions, -top_n)[0, -top_n:]
        top_probabilities = predictions[0, top_indices]
        top_labels = [str(i) for i in top_indices]

        # 返回分类结果
        # results = [{"label": label, "prob": float(prob)} for label, prob in zip(top_labels, top_probabilities)]
        results = [{"label": class_label_to_name[label], "prob": float(prob)} for label, prob in zip(top_labels, top_probabilities)]

        return results


if __name__ == "__main__":
    # 创建 MobileNetClassifier 实例并加载模型
    model_path = "sushi_settlement/models/mobilenet_sushi.h5"
    classifier = CNNClassifier(model_path)

    image = Image.open("/Users/stacy/iss/5002project/backend/tests/images/Shako_Nigiri.jpg")
    top_n = 3  # 获取前三个类别的概率
    results = classifier.predict(image, top_n)
    print(results)