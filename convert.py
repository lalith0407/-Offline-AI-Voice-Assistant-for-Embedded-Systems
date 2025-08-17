import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('ds_cnn.h5')

# Create a TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Optimization for smaller and faster model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the .tflite model
with open('ds_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Successfully converted to ds_cnn.tflite!")
