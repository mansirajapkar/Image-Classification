import tensorflow as tf

# Load your old model (the one that works locally)
old_model = tf.keras.models.load_model("flower_model.keras", compile=False)

# Re-save in a fully compatible format
old_model.save("flower_model_2.keras", save_format="keras")
print("✅ Model re-saved successfully!")
