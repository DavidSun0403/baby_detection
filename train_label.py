import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from utils import training_data_folder, valid_data_folder, write_tfrecord, create_dataset, process_label, process_image_path

# process train data, only need to run at the first time
train_data = pd.read_csv(training_data_folder + '_annotations.csv')
train_image_paths = process_image_path(train_data['filename'], training_data_folder)
train_labels = process_label(train_data['class'])
write_tfrecord(train_image_paths, train_labels, 'result/train.tfrecord')

#  process valid data, only need to run at the first time
valid_data = pd.read_csv(valid_data_folder + '_annotations.csv')
valid_image_paths = process_image_path(valid_data['filename'], valid_data_folder)
valid_labels = process_label(valid_data['class'])
write_tfrecord(valid_image_paths, valid_labels, 'result/valid.tfrecord')

train_dataset = create_dataset('result/train.tfrecord')
val_dataset = create_dataset('result/valid.tfrecord', shuffle=False)

# Use a pre-trained model (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
    tf.keras.layers.Dense(1000, activation='softmax')
])

# Use a constant learning rate
initial_learning_rate = 0.002

# Create optimizer with constant learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Double-check that the learning rate is a float, not a schedule
if not isinstance(optimizer.learning_rate, float):
    print("Warning: Learning rate is not a float. Fixing it now.")
    optimizer.learning_rate = tf.keras.backend.get_value(optimizer.learning_rate)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

model.save("result/baby_label.keras")
