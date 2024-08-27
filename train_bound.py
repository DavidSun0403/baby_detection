import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
from utils import training_data_folder, valid_data_folder, test_data_folder, load_and_preprocess_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch

# Process train data
train_images = []
train_bboxes = []
train_data = pd.read_csv(training_data_folder + '_annotations.csv')
for index, row in train_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = training_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    train_images.append(img)
    train_bboxes.append(bbox_n)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_bboxes = np.array(train_bboxes)

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
datagen.fit(train_images)

# Process validation data
valid_images = []
valid_bboxes = []
valid_data = pd.read_csv(valid_data_folder + '_annotations.csv')
for index, row in valid_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = valid_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    valid_images.append(img)
    valid_bboxes.append(bbox_n)

# Convert lists to numpy arrays
valid_images = np.array(valid_images)
valid_bboxes = np.array(valid_bboxes)

# Process test data
test_images = []
test_bboxes = []
test_data = pd.read_csv(test_data_folder + '_annotations.csv')
for index, row in test_data.iterrows():
    bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
    image_path = test_data_folder + row["filename"]
    img, bbox_n = load_and_preprocess_image(image_path, bbox)
    test_images.append(img)
    test_bboxes.append(bbox_n)

# Convert lists to numpy arrays
test_images = np.array(test_images)
test_bboxes = np.array(test_bboxes)

# Hyperparameter tuning
def build_model(hp):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(hp.Int('units', min_value=64, max_value=256, step=64), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
        layers.Dense(4)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

tuner = RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=2, directory='my_dir', project_name='bbox_tuning')
tuner.search(train_images, train_bboxes, epochs=10, validation_data=(valid_images, valid_bboxes))
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model with learning rate scheduler and early stopping
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

best_model.fit(datagen.flow(train_images, train_bboxes, batch_size=32),
               epochs=50,
               validation_data=(valid_images, valid_bboxes),
               callbacks=[lr_schedule, early_stopping])

# Evaluate the model on the test data
test_loss = best_model.evaluate(test_images, test_bboxes)
print(f"Test loss: {test_loss}")

best_model.save("result/baby_bound.keras")
