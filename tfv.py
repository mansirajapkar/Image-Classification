import tensorflow as tf
print(tf.__version__)

model = tf.keras.models.load_model("flower_model_2.keras")
model.save("oxford_flower_model.h5")