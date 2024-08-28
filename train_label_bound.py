import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import training_data_folder, valid_data_folder, test_data_folder

def load_and_preprocess_image(image_path, bbox):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    bbox = np.array(bbox) / [original_width, original_height, original_width, original_height]  # Normalize bbox
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    return image, bbox

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process train data
train_images = []
train_bboxes = []
train_labels = []
train_data = pd.read_csv(training_data_folder + '_annotations.csv')

for index, row in train_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = training_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    train_images.append(img)
    train_bboxes.append(bbox_n)
    train_labels.append(1 if row['class'] == 'Baby' else 0)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_bboxes = np.array(train_bboxes)
train_labels = np.array(train_labels)

# Process validation data
valid_images = []
valid_bboxes = []
valid_labels = []
valid_data = pd.read_csv(valid_data_folder + '_annotations.csv')

for index, row in valid_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = valid_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    valid_images.append(img)
    valid_bboxes.append(bbox_n)
    valid_labels.append(1 if row['class'] == 'Baby' else 0)

# Convert lists to numpy arrays
valid_images = np.array(valid_images)
valid_bboxes = np.array(valid_bboxes)
valid_labels = np.array(valid_labels)

# Process test data
test_images = []
test_bboxes = []
test_labels = []
test_data = pd.read_csv(test_data_folder + '_annotations.csv')

for index, row in test_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = test_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    test_images.append(img)
    test_bboxes.append(bbox_n)
    test_labels.append(1 if row['class'] == 'Baby' else 0)

# Convert lists to numpy arrays
test_images = np.array(test_images)
test_bboxes = np.array(test_bboxes)
test_labels = np.array(test_labels)

# Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Unfreeze the base model

# Fine-tune from this layer onwards
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define the model
input_layer = layers.Input(shape=(224, 224, 3))
x = base_model(input_layer, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Bounding box output
bbox_output = layers.Dense(128, activation='relu')(x)
bbox_output = layers.Dense(4, name='bbox')(bbox_output)

# Class label output
class_output = layers.Dense(128, activation='relu')(x)
class_output = layers.Dense(1, activation='softmax', name='class')(class_output)

# Combine the outputs
model = models.Model(inputs=input_layer, outputs=[bbox_output, class_output])

# Compile the model with a combined loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'bbox': 'mean_squared_error', 'class': 'sparse_categorical_crossentropy'},
              metrics={'bbox': 'mean_squared_error', 'class': 'accuracy'})
print(len(train_images))
print(len(train_bboxes))
print(len(train_labels))
# Train the model with validation data
model.fit(datagen.flow(train_images, {'bbox': train_bboxes, 'class': train_labels}, batch_size=32),
          epochs=50, validation_data=(valid_images, {'bbox': valid_bboxes, 'class': valid_labels}))

# Evaluate the model on the test data
test_loss, test_bbox_loss, test_class_loss, test_bbox_mse, test_class_acc = model.evaluate(test_images, {'bbox': test_bboxes, 'class': test_labels})
print(f"Test loss: {test_loss}, Test bbox loss: {test_bbox_loss}, Test class loss: {test_class_loss}, Test bbox MSE: {test_bbox_mse}, Test class accuracy: {test_class_acc}")

model.save("result/baby_bound_and_label.keras")
