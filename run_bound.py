import tensorflow as tf
import numpy as np
from utils import load_and_preprocess_image
import cv2

# Load the trained model
model = tf.keras.models.load_model("result/baby_bound.keras")

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]  # Save the original image shape
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize the image
    return image, original_shape


# Function to predict bounding box for a single image and save the image with bbox
def predict_and_save_bounding_box(image_path, output_path):
    image, original_shape = load_and_preprocess_image(image_path)
    img = np.expand_dims(image, axis=0)  # Add batch dimension
    pred_bbox = model.predict(img)[0]

    # Load the original image using OpenCV
    original_img = cv2.imread(image_path)
    height, width, _ = original_img.shape

    # Convert normalized bbox to actual coordinates
    xmin = int(pred_bbox[0] * width)
    ymin = int(pred_bbox[1] * height)
    xmax = int(pred_bbox[2] * width)
    ymax = int(pred_bbox[3] * height)

    # Draw the bounding box on the image
    cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Save the image with bounding box
    cv2.imwrite(output_path, original_img)

# Example usage
image_path = "cutest_baby.jpg"
output_path = "cutest_baby_with_bbox.jpg"
predict_and_save_bounding_box(image_path, output_path)
print(f"Image with predicted bounding box saved to {output_path}")
