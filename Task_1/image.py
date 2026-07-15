import tensorflow as tf
import numpy as np

# Load pre-trained model (no training needed, downloads automatically)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load any image
img = tf.keras.utils.load_img("photo.jpg", target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[tf.newaxis])

# Predict
predictions = model.predict(img_array)
results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
for name, label, score in results[0]:
    print(f"{label}: {score*100:.1f}%")