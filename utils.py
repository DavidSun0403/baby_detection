from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_data_folder  = "data/baby_detection_data/train/"
valid_data_folder  = "data/baby_detection_data/valid/"
test_data_folder  = "data/baby_detection_data/test/"

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)


# Function to create a tf.train.Example from image and label
def create_example(image_path, label):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to a fixed size
    image_bytes = image.tobytes()
    
    feature = {
        'image': _bytes_feature(image_bytes),
        'label': _int64_feature(label)
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Function to convert image to bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Function to convert integer to feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Function to create a tf.train.Example from image and label
def create_example(image_path, label):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to a fixed size
    image_bytes = image.tobytes()
    
    feature = {
        'image': _bytes_feature(image_bytes),
        'label': _int64_feature(label)
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Function to write TFRecord file
def write_tfrecord(image_paths, labels, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path, label in zip(image_paths, labels):
            example = create_example(image_path, label)
            writer.write(example.SerializeToString())

# Function to parse TFRecord example
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    label = example['label']
    return image, label

def create_dataset(tfrecord_file, batch_size=32, shuffle=True, augment=False):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord_fn)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    if augment:
        dataset = dataset.map(lambda x, y: (datagen.flow(x, y, batch_size=batch_size)[0]))
    return dataset

def process_label(input_list):
    return [1 if element == 'baby' else 0 for element in input_list]

def process_image_path(input_list, prefix):
    return [prefix + element for element in input_list]
