import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("result/improved_baby_face_detection.h5")

def predict_and_save(image_path, output_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Predict bounding box
    predicted_bbox = model.predict(image_array)[0]
    #predicted_bbox = np.array([0.38125, 0.296875, 0.51875, 0.4953125])
    print(predicted_bbox)
    # Denormalize bounding box
    h, w, _ = image.shape
    predicted_bbox = predicted_bbox * [w, h, w, h]

    # Draw bounding box on the image
    x_min, y_min, x_max, y_max = map(int, predicted_bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Save the image with the bounding box
    cv2.imwrite(output_path, image)

# Example usage
predict_and_save('HJK-7.jpg', 'Crybaby-921_with_bbox.jpg')