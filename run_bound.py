import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
from keras.saving import register_keras_serializable

@register_keras_serializable()
def combined_loss(y_true, y_pred, beta=1.0, alpha=0.25, gamma=2.0):
    diff = tf.abs(y_true - y_pred)
    less_than_beta = tf.less(diff, beta)
    loss = tf.where(less_than_beta, 0.5 * tf.square(diff) / beta, diff - 0.5 * beta)
    smooth_l1_loss = tf.reduce_mean(loss)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    loss = weight * cross_entropy
    focal_loss = tf.reduce_mean(loss)
    return smooth_l1_loss + focal_loss

# Load the model with custom objects
custom_objects = {'combined_loss': combined_loss}
model = load_model('result/improved_baby_face_detection.keras', custom_objects=custom_objects, safe_mode=False)

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
predict_and_save('cutest_baby.jpg', 'Crybaby-921_with_bbox.jpg')