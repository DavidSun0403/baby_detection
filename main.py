import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def detect_baby(image_path):
    # Load the pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=5)[0]

    # Check if 'baby' is in the top 5 predictions
    for _, label, score in decoded_preds:
        print(label, score)
        if 'baby' in label:
            return True, score

    return False, 0.0

# Example usage
image_path = 'tree.jpg'
has_baby, confidence = detect_baby(image_path)

if has_baby:
    print(f"Baby detected with {confidence:.2%} confidence.")
else:
    print("No baby detected in the image.")
