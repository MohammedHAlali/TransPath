'''
This is for binary classification
class 0 includes adc and scc
class 1 includes normal
'''

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import argparse


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# Define the command-line argument parser
parser = argparse.ArgumentParser(description='Training a CNN model with variable batch size.')

# Add the batch_size argument
parser.add_argument('--batch_size', type=int, default=2, help='Size of each batch for training.')
# Adjust according to your memory and computational power
parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train")
# Parse the command-line arguments
args = parser.parse_args()

# Use the parsed batch_size argument
batch_size = args.batch_size
epochs = args.epochs
print(f'Using batch size: {batch_size}')
print(f'Using epoch of: {epochs}')

# Define the total dataset size and class distribution
class_0_size = 101
class_1_size = 107
class_2_size = 103
dataset_size = class_0_size + class_1_size + class_2_size  # Total number of elements

# define the total validation dataset size:
adc_val_size = 20
scc_val_size = 9
nor_val_size = 15
val_dataset_size = adc_val_size + scc_val_size + nor_val_size

# Generator function definition with additional debug information
def class_data_generator(file, label, batch_size):
    """
    Yields batches of data for a specific class and its associated label.
    
    Args:
    - file: Path to the numpy file.
    - label: Integer label for the class.
    - batch_size: Number of samples per batch.
    
    Yields:
    - x: Batch of inputs of shape (batch_size, 40911, 1024, 1).
    - y: Corresponding labels for the batch.
    """
    data = np.load(file)
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=3)  # Expand to (samples, 40911, 1024, 1)
    #print(f'from path={file}, loaded numpy array of shape={data.shape}')
    #print(f'Label assigned for this dataset: {label}')
    
    inputs = data  # Use the loaded array directly
    while True:
        for i in range(0, len(inputs), batch_size):
            x = inputs[i:i + batch_size]
            y = np.full((len(x),), label, dtype=np.int32)
            yield x, y

# File paths and labels for each class (replace with your actual file paths)
# training files

adc_train_file = 'features/train/adc/x_adc.npy'
scc_train_file = 'features/train/scc/x_scc.npy'
nor_train_file = 'features/train/normal/x_normal.npy'
fv_len = 768

# add function that takes three strings for filepath
# each string is for one class data
# the function returns tf.data.dataset
#this function should be used for adc, scc, nor in all three test datasets
def prepare_data(adc_file, scc_file, nor_file, batch_size, phase):
    print('================== processing {} dataset ==============='.format(phase))
    # Create testing1 datasets for each class
    adc_dataset = tf.data.Dataset.from_generator(
    lambda: class_data_generator(adc_file, 0, batch_size),  # Adjust batch size as needed
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,)))
    
    scc_dataset = tf.data.Dataset.from_generator(
    lambda: class_data_generator(scc_file, 0, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,)))

    nor_dataset = tf.data.Dataset.from_generator(
    lambda: class_data_generator(nor_file, 1, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,)))

    # Print elements from each dataset to ensure correct label association
    #print("Checking dataset adc (class 0):")
    #for x, y in adc_dataset.take(2):  # Take 2 samples to inspect
    #    print(f"Input shape: {x.shape}, Label: {y.numpy()}")

    #print("Checking dataset scc (class 1):")
    #for x, y in scc_dataset.take(2):  # Take 2 samples to inspect
    #    print(f"Input shape: {x.shape}, Label: {y.numpy()}")

    #print("Checking dataset normal (class 2):")
    #for x, y in nor_dataset.take(2):  # Take 2 samples to inspect
    #    print(f"Input shape: {x.shape}, Label: {y.numpy()}")

    # Combine test datasets using zip
    my_dataset = tf.data.Dataset.zip((adc_dataset, scc_dataset, nor_dataset))
    print('my dataset: ', my_dataset)
    # Flatten the combined dataset by mapping to a unified structure
    my_dataset = my_dataset.flat_map(
        lambda d0, d1, d2: tf.data.Dataset.from_tensors(d0).concatenate(
        tf.data.Dataset.from_tensors(d1)).concatenate(tf.data.Dataset.from_tensors(d2)))
    print('my dataset: ', my_dataset)

    my_dataset = my_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # Inspect the combined dataset after shuffle and repeat
    print("Inspecting the dataset")
    for i, (x, y) in enumerate(my_dataset.take(10)):
        print(f"Batch {i}: Label shape: {y.shape}. Labels = {y.numpy()}")
    return my_dataset


# Create a dataset for each class
dataset_0 = tf.data.Dataset.from_generator(
    lambda: class_data_generator(adc_train_file, 0, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,))
)

dataset_1 = tf.data.Dataset.from_generator(
    lambda: class_data_generator(scc_train_file, 0, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,))
)

dataset_2 = tf.data.Dataset.from_generator(
    lambda: class_data_generator(nor_train_file, 1, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,))
)

# Combine datasets using zip
combined_dataset = tf.data.Dataset.zip((dataset_0, dataset_1, dataset_2))
print('combined dataset: ', combined_dataset)
# Flatten the combined dataset by mapping to a unified structure
combined_dataset = combined_dataset.flat_map(
    lambda d0, d1, d2: tf.data.Dataset.from_tensors(d0).concatenate(
        tf.data.Dataset.from_tensors(d1)).concatenate(tf.data.Dataset.from_tensors(d2))
)
print('combined dataset: ', combined_dataset)

# Shuffle the combined dataset with the buffer size equal to the total dataset size
combined_dataset = combined_dataset.shuffle(buffer_size=dataset_size)

# Repeat and prefetch for performance
combined_dataset = combined_dataset.repeat()
combined_dataset = combined_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#validation files
adc_val_file = 'features/val/adc/x_adc.npy'
scc_val_file = 'features/val/scc/x_scc.npy'
nor_val_file = 'features/val/normal/x_normal.npy'
val_dataset = prepare_data(adc_val_file, scc_val_file, nor_val_file, batch_size=1, phase='val')


# Create a Sequential model
model = models.Sequential()

# First convolutional layer with input shape (40911, 1024, 1)
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(40911, fv_len, 1)))
model.add(layers.MaxPooling2D((2, 2)))  # Reduce the size

model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # Reduce the size

# Second convolutional layer
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # Further reduce the size

# Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to feed into Dense layers
model.add(layers.Flatten())

# Fully connected Dense layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

# Output layer with softmax activation for classification
model.add(layers.Dense(1, activation='sigmoid'))  # Adjust output units according to your number of classes

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary to inspect the architecture
model.summary()


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            for weight in layer.weights:
                size = tf.size(weight).numpy() * weight.dtype.size
                print(f"Layer {layer.name} - Weight {weight.name}: Shape = {weight.shape}, Memory = {size / (1024 ** 2):.2f} MB")

# Attach the callback to your training
memory_callback = MemoryUsageCallback()


print('batch_size = ', batch_size)
steps = dataset_size//batch_size
print('training for num of steps = ', steps)
val_steps = val_dataset_size


import matplotlib.pyplot as plt
import datetime

# Function to plot and save training history with a timestamped filename
def plot_training_history(history):
    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'training_performance_{timestamp}.png'  # Filename with timestamp

    # Create a figure for loss and accuracy plots
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot as a PNG file with the timestamped filename
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    print(f"Training history plot saved as: {save_path}")

# Example usage after training
#history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
from tensorflow.keras.callbacks import EarlyStopping

# Define Early Stopping callback
early_stopping = EarlyStopping(
    verbose=1,
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=epochs//6,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the model weights from the epoch with the best value
)
print('using early stopping: ', early_stopping)

history = model.fit(combined_dataset,
        validation_data=val_dataset,
        validation_steps=val_steps,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=[early_stopping])
plot_training_history(history)

# Save the model as an HDF5 file
# Generate a timestamped filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f'my_model_{timestamp}.h5'  # Filename with timestamp
model.save(save_path)  # Use .h5 extension for HDF5 format
print('Trained model saved successfully in path: ', save_path)

######## prepare testing ########################

def load_data_directly(adc_file, scc_file, nor_file, phase):
    print('======= loading {} dataset ======'.format(phase))
    # Step 1: Load data from numpy files
    adc_test = np.load(adc_file)
    scc_test = np.load(scc_file)
    nor_test = np.load(nor_file)
    print('loaded data of shape, adc={}, scc={}, normal={}'.format(adc_test.shape, scc_test.shape, nor_test.shape))
    # Step 2: Create labels for binary classification
    # ADC = 0, SCC = 0 (both cancerous), Normal = 1 (non-cancerous)
    adc_labels = np.zeros(adc_test.shape[0], dtype=np.int32)    # Label 0 for ADC (cancerous)
    scc_labels = np.zeros(scc_test.shape[0], dtype=np.int32)    # Label 0 for SCC (cancerous)
    nor_labels = np.ones(nor_test.shape[0], dtype=np.int32)     # Label 1 for Normal (non-cancerous)

    # Step 3: Concatenate the data and labels
    X_data = np.concatenate([adc_test, scc_test, nor_test], axis=0)
    y_labels = np.concatenate([adc_labels, scc_labels, nor_labels], axis=0)
    print('test dataset in numpy array of shape x={}, y={}'.format(X_data.shape, y_labels.shape))

    # Step 5: Create a tf.data.Dataset from the numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((X_data, y_labels))
    print('testing dataset in tf.data: ', dataset)

    dataset = dataset.batch(1)
    return dataset


import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv


def evaluate_model(test_dataset, dataset_name="dataset"):
    """
    Evaluates a trained model on a test dataset, computes metrics, and saves the results.

    Parameters:
    - model: Trained TensorFlow model
    - test_dataset: tf.data.Dataset, Test dataset (features and labels)
    - dataset_name: str, Name of the dataset for the results file (e.g., 'adc_test1', 'scc_test1', etc.)

    Returns:
    - None
    """
    
    # Step 1: Get the true labels from the test dataset
    y_true = np.concatenate([y for _, y in test_dataset], axis=0)
    # Step 2: Get the predicted labels from the model (make predictions on the whole dataset)
    X_test = np.concatenate([X for X, _ in test_dataset], axis=0)
    y_pred_prob = model.predict(X_test, batch_size=2)  # Predict probabilities
    y_pred = tf.round(y_pred_prob).numpy().astype(int)  # Convert probabilities to 0/1

    # Step 3: Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Step 4: Confusion Matrix for False Positives and False Negatives
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {dataset_name}')

    # Step 5: Save the confusion matrix and metrics with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'{dataset_name}_evaluation_{timestamp}.png'

    # Save the plot
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

    # Step 6: Print the metrics
    print(f"Results for {dataset_name}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Confusion matrix saved as: {filename}")
    
    # Step 7: Save metrics to a CSV file
    csv_filename = f'{dataset_name}_evaluation_{timestamp}.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{accuracy:.3f}"])
        writer.writerow(["Precision", f"{precision:.3f}"])
        writer.writerow(["Recall", f"{recall:.3f}"])
        writer.writerow(["F1-score", f"{f1:.3f}"])
        writer.writerow(["False Positives", fp])
        writer.writerow(["False Negatives", fn])
    
    print(f"Metrics saved as CSV file: {csv_filename}")



adc_test1 = 'features/test1/adc/x_adc.npy'
scc_test1 = 'features/test1/scc/x_scc.npy'
nor_test1 = 'features/test1/normal/x_normal.npy'

adc_test2 = 'features/test2/adc/x_adc.npy'
scc_test2 = 'features/test2/scc/x_scc.npy'
nor_test2 = 'features/test2/normal/x_normal.npy'

adc_test3 = 'features/test3/adc/x_adc.npy'
scc_test3 = 'features/test3/scc/x_scc.npy'
nor_test3 = 'features/test3/normal/x_normal.npy'


with tf.device('/CPU:0'):
    test1_dataset = load_data_directly(adc_test1, scc_test1, nor_test1, 'test1')
    test2_dataset = load_data_directly(adc_test2, scc_test2, nor_test2, 'test2')
    test3_dataset = load_data_directly(adc_test3, scc_test3, nor_test3, 'test3')
    evaluate_model(test1_dataset, dataset_name="test1")
    evaluate_model(test2_dataset, dataset_name="test2")
    evaluate_model(test3_dataset, dataset_name="test3")



#test1_dataset = prepare_data(adc_test1, scc_test1, nor_test1, 1, 'test1')

#test2_dataset = prepare_data(adc_test2, scc_test2, nor_test2, 1, 'test2')

#test3_dataset = prepare_data(adc_test3, scc_test3, nor_test3, 1, 'test3')

print('done')
