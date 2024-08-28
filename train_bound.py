import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Lambda
from keras.saving import register_keras_serializable

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

def load_and_preprocess_image(image_path, bbox):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize bbox
    bbox = np.array(bbox) / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    
    # Data augmentation
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        bbox[0], bbox[2] = 1 - bbox[2], 1 - bbox[0]
    
    image = cv2.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    
    return image, bbox

def load_dataset(data_folders, annotations_file, ignore_not_baby=False):
    images = []
    bboxes = []
    for data_folder in data_folders: 
        data = pd.read_csv(data_folder + annotations_file)
        for index, row in data.iterrows():
            if ignore_not_baby and row["class"] != 'Baby':
                continue
            if isinstance(row["filename"], str):
                bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                image_path = data_folder + row["filename"]
                img, bbox_n = load_and_preprocess_image(image_path, bbox)
                images.append(img)
                bboxes.append(bbox_n)
    return np.array(images), np.array(bboxes)

# Load and preprocess data
all_images, all_bboxes = load_dataset(["data/baby_detection_data/train/", "data/baby_detection_data2/train/", 
                                       "data/baby_detection_data/valid/", "data/baby_detection_data2/valid/"], '_annotations.csv')

# Split the data
train_images, val_images, train_bboxes, val_bboxes = train_test_split(all_images, all_bboxes, test_size=0.2, random_state=42)

test_images, test_bboxes = load_dataset(["data/baby_detection_data/test/", "data/baby_detection_data2/test/"], '_annotations.csv')

print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(val_images)}")
print(f"Test samples: {len(test_images)}")

# Build the model
base_model = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

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

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4),  # Remove sigmoid activation
    Lambda(lambda x: tf.clip_by_value(x, 0, 1))  # Clip values between 0 and 1
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = combined_loss  # Use the custom loss function
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_images, train_bboxes,
    validation_data=(val_images, val_bboxes),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(test_images, test_bboxes)
print(f"Test loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Save the model
model.save("result/improved_baby_face_detection.keras")