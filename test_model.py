import tensorflow as tf
import pandas as pd
from utils import test_data_folder, write_tfrecord, create_dataset, process_label, process_image_path


# process test data, only need to run at the first time
test_data = pd.read_csv(test_data_folder + '_annotations.csv')
test_image_paths = process_image_path(test_data['filename'], test_data_folder)
test_labels = process_label(test_data['class'])
write_tfrecord(test_image_paths, test_labels, 'result/test.tfrecord')

test_dataset = create_dataset('result/test.tfrecord', shuffle=False)

model = tf.keras.models.load_model('result/baby.keras')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f} Test loss: {test_loss:.4f}")