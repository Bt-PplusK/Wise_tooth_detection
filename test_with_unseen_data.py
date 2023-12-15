# -*- coding: utf-8 -*-
"""Test_with_Unseen_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PU8u6AhTUuvcGPq276_FO60o8lXpxXK9
"""

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import tensorflow as tf

filename = "/content/drive/MyDrive/final project/data reservoir/3K2X 5.9.2023/Test/class A, (+) on both sides/209.jpg"

image = cv2.imread(filename)
print(image.shape)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image,(224,224))
image = np.expand_dims([image], axis=-1)
print(image.shape)

model = tf.keras.models.load_model('/content/drive/MyDrive/my_model4.h5')
pred = model.predict(image)
print(pred)