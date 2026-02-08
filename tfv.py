import tensorflow as tf

model = tf.keras.models.load_model("oxford_flower_model.h5")
model.save("flower_model.keras")   # 👈 NEW FORMAT (this fixes everything)
print("done")