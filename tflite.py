# -*- coding: utf-8 -*-
"""tflite.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ihxm17Kt2I8tqAspEWMy2izCxNolgqvt
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf

model = tf.keras.models.load_model("/content/drive/MyDrive/my_model4.h5")
model.save("saved_model")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)