import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('result/baby.keras')

# Load and preprocess the image
img_path = 'cutest_baby.jpg'
img = image.load_img(img_path, target_size=(224, 224)) 
img_array = image.img_to_array(img)
img_array = img_array / 255.0 
img_array = np.expand_dims(img_array, axis=0) 

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class[0]}")