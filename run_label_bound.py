import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def predict_and_save(image_path, output_path):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Predict bounding box and class label
    predicted_bbox, predicted_class = model.predict(image_array)
    predicted_bbox = predicted_bbox[0]
    predicted_class = np.argmax(predicted_class[0])

    # Denormalize bounding box
    predicted_bbox = predicted_bbox * [original_width, original_height, original_width, original_height]

    # Draw bounding box and label on the image
    x_min, y_min, x_max, y_max = map(int, predicted_bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    label = label_encoder.inverse_transform([predicted_class])[0]
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the image with the bounding box and label
    cv2.imwrite(output_path, image)

# Example usage
predict_and_save('cutest_baby.jpg', 'cutest_baby_label_bound.jpg')