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
import tensorflow as tf
from PIL import Image
import numpy as np

class_label_to_name = {
    '0': 'Abokado_Maki',
    '1': 'Aji',
    '2': 'Akagai',
    '3': 'Akami',
    '4': 'Amaebi',
    '5': 'Ankimo',
    '6': 'Aoyagi',
    '7': 'Awabi',
    '8': 'Battera_Sushi',
    '9': 'California_Roll',
    '10': 'Daikon_Oshinko_Maki',
    '11': 'Ebi_Nigiri',
    '12': 'Funazushi',
    '13': 'Futomaki',
    '14': 'Gyu_Nigiri',
    '15': 'Hamaguri',
    '16': 'Hamo',
    '17': 'Hirame_&_Karei',
    '18': 'Hokkigai',
    '19': 'Hotate_Nigiri',
    '20': 'Ika_Nigiri',
    '21': 'Ikura_Gunkan',
    '22': 'Inarizushi',
    '23': 'Kamaboko_Kani',
    '24': 'Kanpyomaki',
    '25': 'Kappa_Maki',
    '26': 'Kazunoko',
    '27': 'Kohada',
    '28': 'Kuruma',
    '29': 'Meharizushi',
    '30': 'Mirugai',
    '31': 'Natto_Maki',
    '32': 'Negitoro',
    '33': 'Saba',
    '34': 'Sake_Nigiri',
    '35': 'Sayori',
    '36': 'Shako_Nigiri',
    '37': 'Shirasu',
    '38': 'Shiroebi',
    '39': 'Tai_&_Madai',
    '40': 'Tako_Nigiri',
    '41': 'Tamagoyaki',
    '42': 'Tekkamaki',
    '43': 'Temaki',
    '44': 'Temarizushi',
    '45': 'Tobiko_Nigiri',
    '46': 'Torigai',
    '47': 'Toro',
    '48': 'Unagi',
    '49': 'Uni'
}

class MobileNetClassifier:
    def __init__(self, model_path):
        # 加载 MobileNet 模型
        self.model = tf.keras.models.load_model(model_path)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.training = False
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        return

    def predict(self, img, top_n=3):
        # 将图像转换为3通道的彩色图像
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((224, 224))  # 调整图像大小以符合 MobileNet 的输入要求
        img = np.asarray(img)
        #img = img / 255.0  # 归一化像素值
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
        results.sort(key=lambda x: x["prob"], reverse=True)
        return results


if __name__ == "__main__":
    # 创建 MobileNetClassifier 实例并加载模型
    model_path = "sushi_settlement/models/mobilenet_sushi.h5"
    classifier = MobileNetClassifier(model_path)

    image = Image.open("/Users/stacy/iss/5002project/backend/tests/images/Temarizushi.jpg")
    top_n = 3  # 获取前三个类别的概率
    results = classifier.predict(image, top_n)
    print(results)