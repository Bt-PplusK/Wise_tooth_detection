# -*- coding: utf-8 -*-
"""Using_the_model_with_test_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ej-5iO1Zmri0BnHUP6ow30PAnSUg0uOn
"""

from google.colab import drive
drive.mount('/content/drive')

test_dir = '/content/drive/MyDrive/final project/data reservoir/3K2X 5.9.2023/Test'


datagen_test = ImageDataGenerator(rescale=1.0/255,
                                   fill_mode='nearest'
                                   )
test_set = datagen_test.flow_from_directory(
                                            test_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            class_mode="categorical",
                                            shuffle=False)

model = keras.models.load_model('/content/drive/MyDrive/my_model4.h5')

model.evaluate(test_set)