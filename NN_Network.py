#Bibek Kumar Sharma
#cs455

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

# Load and preprocess the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the images to improve model convergence
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape data for FCNN (flatten 28x28 images to 784-dimensional vectors)
train_images_flat = train_images.reshape(-1, 28 * 28)
test_images_flat = test_images.reshape(-1, 28 * 28)

# Reshape data for CNN (add channel dimension for grayscale images)
train_images_cnn = train_images.reshape(-1, 28, 28, 1)
test_images_cnn = test_images.reshape(-1, 28, 28, 1)

### Fully Connected Neural Network (FCNN) Model
fcnn_model = Sequential([
    Input(shape=(784,)),  # Use Input layer to avoid warnings about input_shape
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

fcnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the FCNN model with a reduced batch size to manage memory usage
fcnn_history = fcnn_model.fit(train_images_flat, train_labels, epochs=20, batch_size=16,
                              validation_data=(test_images_flat, test_labels))

### Convolutional Neural Network (CNN) Model
cnn_model = Sequential([
    Input(shape=(28, 28, 1)),  # Use Input layer to avoid warnings about input_shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with a reduced batch size to manage memory usage
cnn_history = cnn_model.fit(train_images_cnn, train_labels, epochs=20, batch_size=16,
                            validation_data=(test_images_cnn, test_labels))

### Plotting Function
def plot_metrics(history, model_type):
    epochs = range(len(history.history['accuracy']))
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_type} - Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_type} - Loss')

    plt.show()

# Plot metrics for both models
plot_metrics(fcnn_history, "FCNN")
plot_metrics(cnn_history, "CNN")
