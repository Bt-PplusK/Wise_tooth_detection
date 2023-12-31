# -*- coding: utf-8 -*-
"""Wise_tooth_detection_3K2X.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1imlMQVqhI0oit9hFxU47HwIERF7mgkcN
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
import cv2
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
device_name = tf.test.gpu_device_name()

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')

"""# Data Preprocessing"""

train_dir = '/content/drive/MyDrive/final project/data reservoir/3K2X 5.9.2023/train'
val_dir ='/content/drive/MyDrive/final project/data reservoir/3K2X 5.9.2023/val'

import cv2
image = cv2.imread("/content/drive/MyDrive/final project/data reservoir/output/train/class B, (-) bilaterally or unilaterally/abul (1).jpg")

import matplotlib.pyplot as plt
plt.imshow(image)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

batch_size = 32
train_set = datagen_train.flow_from_directory(
                                              train_dir,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              color_mode='grayscale',
                                              class_mode="categorical",
                                              shuffle=True)

datagen_val = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)


val_set = datagen_val.flow_from_directory(
                                            val_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            class_mode="categorical",
                                            shuffle=False)

no_images = train_set.samples
print(no_images)

"""# Modeling"""

import tensorflow as tf
from tensorflow import keras

input_shape = (224, 224, 1)

dropout_rate = 0.3

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(dropout_rate),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(dropout_rate),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(dropout_rate),

    keras.layers.Flatten(),

    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(dropout_rate),

    keras.layers.Dense(2, activation="softmax")
])

learning_rate = 0.0005
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

checkpoint = keras.callbacks.ModelCheckpoint(
    'model_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

num_of_training_images = no_images
history = model.fit(train_set,callbacks=[checkpoint],
                    steps_per_epoch=num_of_training_images//batch_size,
                    epochs=10,
                    verbose = 2,
                    shuffle = False)

import matplotlib.pyplot as plt
acc = history.history['accuracy']

loss = history.history['loss']

epochs_range = range(1, 11)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

class_dictionary = val_set.class_indices
print('Labels dictionary',class_dictionary)

y_predicted = model.predict(val_set)

y_pred = []
for i in  y_predicted:
  print(i)
  print(np.argmax(i))
  print()
  y_pred.append(np.argmax(i))

y_val = val_set.classes.tolist()

"""# Evaluation"""

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_val,y_pred)
print('confusion_matrix')
print(confusion_matrix)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print('Accuracy Score',accuracy_score(y_val,y_pred)*100,'%')
print('Precision Macro Score ',precision_score(y_val,y_pred,average = 'macro')*100,'%')
print('Recall_Score',recall_score(y_val,y_pred, average = 'macro')*100,'%')
print('F1_Score',f1_score(y_val,y_pred, average = 'macro')*100,'%')

"""# Save model"""

model.save('my_model4.h5')