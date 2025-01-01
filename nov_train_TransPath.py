'''
Training our classifier on feature vectors that were extracted from CTransPath
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
val_batch_size = 1

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
#validation files
adc_val_file = 'features/val/adc/x_adc.npy'
scc_val_file = 'features/val/scc/x_scc.npy'
nor_val_file = 'features/val/normal/x_normal.npy'

labels = [0, 1, 2]  # Class labels for each file
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
    lambda: class_data_generator(scc_file, 1, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,)))

    nor_dataset = tf.data.Dataset.from_generator(
    lambda: class_data_generator(nor_file, 2, batch_size),
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
    lambda: class_data_generator(scc_train_file, 1, batch_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 40911, fv_len, 1), (None,))
)

dataset_2 = tf.data.Dataset.from_generator(
    lambda: class_data_generator(nor_train_file, 2, batch_size),
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
#val_dataset = val_dataset.repeat() # no repeat on validation dataset

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
model.add(layers.Dense(3, activation='softmax'))  # Adjust output units according to your number of classes

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
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
print('dataset size: ', dataset_size)
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
    patience=epochs//5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the model weights from the epoch with the best value
)

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
    print(' loading {} dataset'.format(phase))
    # Step 1: Load data from numpy files
    adc_test = np.load(adc_file)
    scc_test = np.load(scc_file)
    nor_test = np.load(nor_file)
    print('loaded data of shape, adc={}, scc={}, normal={}'.format(adc_test.shape, scc_test.shape, nor_test.shape)) 
    # Create labels for each class
    adc_labels = np.zeros(adc_test.shape[0])  # Label 0 for ADC
    scc_labels = np.ones(scc_test.shape[0])   # Label 1 for SCC
    nor_labels = np.full(nor_test.shape[0], 2)  # Label 2 for Normal
    # Create TensorFlow datasets for test1
    test_data = np.concatenate([adc_test, scc_test, nor_test], axis=0)
    test_labels = np.concatenate([adc_labels, scc_labels, nor_labels], axis=0)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    print('created dataset: ', test_dataset)
    test_dataset = test_dataset.batch(2)
    return test_dataset



adc_test1 = 'features/test1/adc/x_adc.npy'
scc_test1 = 'features/test1/scc/x_scc.npy'
nor_test1 = 'features/test1/normal/x_normal.npy'
adc_test1_size = 16
scc_test1_size = 19
nor_test1_size = 11
test1_size = adc_test1_size + scc_test1_size + nor_test1_size

adc_test2 = 'features/test2/adc/x_adc.npy'
scc_test2 = 'features/test2/scc/x_scc.npy'
nor_test2 = 'features/test2/normal/x_normal.npy'
adc_test2_size = 17
scc_test2_size = 21
nor_test2_size = 16
test2_size = adc_test2_size + scc_test2_size + nor_test2_size

adc_test3 = 'features/test3/adc/x_adc.npy'
scc_test3 = 'features/test3/scc/x_scc.npy'
nor_test3 = 'features/test3/normal/x_normal.npy'
adc_test3_size = 9
scc_test3_size = 15
nor_test3_size = 9
test3_size = adc_test3_size + scc_test3_size + nor_test3_size

with tf.device('/CPU:0'):
    test1_dataset = load_data_directly(adc_test1, scc_test1, nor_test1, 'test1')
    test2_dataset = load_data_directly(adc_test2, scc_test2, nor_test2, 'test2')
    test3_dataset = load_data_directly(adc_test3, scc_test3, nor_test3, 'test3')



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# Save Evaluation Metrics as PNG
def save_evaluation_metrics(accuracy, precision, recall, f1, save_path):
    """Save evaluation metrics (accuracy, precision, recall, F1) as a PNG image."""
    # Create a DataFrame to store metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    }).round(3)

    # Save metrics to a CSV file (optional)
    metrics_df.to_csv(save_path.replace('.png', '.csv'), index=False)

    # Plot the evaluation metrics DataFrame and save as PNG
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Scale table for better visualization

    # Save as PNG file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Evaluation metrics saved as {save_path}")

# Save Classification Report as PNG
def save_classification_report(y_true, y_pred, target_names, save_path):
    """Generate and save a classification report as a PNG image."""
    # Create classification report DataFrame
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Save the classification report to a CSV file (optional)
    report_df.to_csv(save_path.replace('.png', '.csv'), index=True)

    # Plot the classification report and save as PNG
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    table = ax.table(cellText=report_df.round(2).values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save as PNG file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Classification report saved as {save_path}")



print('Example: Evaluate test1 dataset')
# Call model.predict() on the entire test dataset at once
predictions = model.predict(test1_dataset, verbose=1, steps=test1_size)
print('done predictions of shape: ', predictions.shape)

# Get true labels and predicted labels in one go
y_true_test1 = np.concatenate([y for x, y in test1_dataset], axis=0)
y_pred_test1 = np.argmax(predictions, axis=1)

print('Calculate metrics')
accuracy_test1 = accuracy_score(y_true_test1, y_pred_test1)
precision_test1 = precision_score(y_true_test1, y_pred_test1, average='weighted')
recall_test1 = recall_score(y_true_test1, y_pred_test1, average='weighted')
f1_test1 = f1_score(y_true_test1, y_pred_test1, average='weighted')

print('Save metrics and classification report as PNG')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_evaluation_metrics(accuracy_test1, 
        precision_test1, 
        recall_test1, 
        f1_test1, 
        save_path=f'evaluation_metrics_test1_{timestamp}.png')
save_classification_report(y_true_test1, 
        y_pred_test1, 
        target_names=['ADC', 'SCC', 'Normal'], 
        save_path=f'classification_report_test1_{timestamp}.png')


print(' Example: Evaluate test2 dataset')
predictions = model.predict(test2_dataset, verbose=1, steps=test2_size)
print('done predictions of shape: ', predictions.shape)

# Get true labels and predicted labels in one go
y_true_test2 = np.concatenate([y for x, y in test2_dataset], axis=0)
y_pred_test2 = np.argmax(predictions, axis=1)

print('Calculate metrics')
accuracy_test2 = accuracy_score(y_true_test2, y_pred_test2)
precision_test2 = precision_score(y_true_test2, y_pred_test2, average='weighted')
recall_test2 = recall_score(y_true_test2, y_pred_test2, average='weighted')
f1_test2 = f1_score(y_true_test2, y_pred_test2, average='weighted')

print('Save metrics and classification report as PNG')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_evaluation_metrics(accuracy_test2,
        precision_test2,
        recall_test2,
        f1_test2,
        save_path=f'evaluation_metrics_test2_{timestamp}.png')
save_classification_report(y_true_test2,
        y_pred_test2,
        target_names=['ADC', 'SCC', 'Normal'],
        save_path=f'classification_report_test2_{timestamp}.png')

print(' Example: Evaluate test3 dataset')
predictions = model.predict(test3_dataset, verbose=1, steps=test3_size)
print('done predictions of shape: ', predictions.shape)

# Get true labels and predicted labels in one go
y_true_test3 = np.concatenate([y for x, y in test3_dataset], axis=0)
y_pred_test3 = np.argmax(predictions, axis=1)


print('Calculate metrics')
accuracy_test3 = accuracy_score(y_true_test3, y_pred_test3)
precision_test3 = precision_score(y_true_test3, y_pred_test3, average='weighted')
recall_test3 = recall_score(y_true_test3, y_pred_test3, average='weighted')
f1_test3 = f1_score(y_true_test3, y_pred_test3, average='weighted')

print('Save metrics and classification report as PNG')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_evaluation_metrics(accuracy_test3,
        precision_test3,
        recall_test3,
        f1_test3,
        save_path=f'evaluation_metrics_test3_{timestamp}.png')
save_classification_report(y_true_test3,
        y_pred_test3,
        target_names=['ADC', 'SCC', 'Normal'],
        save_path=f'classification_report_test3_{timestamp}.png')


print('done')
