import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
from PIL import Image
import pathlib
from keras.models import Sequential 
from tensorflow import keras 
from keras import layers 

model = tf.keras.models.load_model('flower_model.keras')
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

img_path = "Rose.jpg"
img = tf.keras.utils.load_img(img_path, target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
  "This image most likely belongs to {} with a {:.2f}% confidence."
  .format(class_names[np.argmax(score)], 100 * np.max(score))
)