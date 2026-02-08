import tensorflow as tf

# Load your OLD model (path to existing model)
old_model_path = "flower_model.keras"

model = tf.keras.models.load_model(old_model_path, compile=False)

# Save again in safe format
model.save("model.h5")

print("✅ Model saved successfully as model.h5")

