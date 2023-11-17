## use this script to train the model
import os
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from google.colab import drive
from sklearn.utils import shuffle
import itertools

# Mount Google Drive
##add your home directory here if working with google colab

# Define image dimensions
img_height, img_width = 279, 181
num_classes = 2  # Number of classes (hamburgers and no hamburgers)

# Define the paths to your dataset directories
hamburgers_dir = "directory of hamburgers folder here"
no_hamburgers_dir = "directory of no hamburgers folder here""

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    images = []
    labels = []

    if directory == hamburgers_dir:
        label = 1
    elif directory == no_hamburgers_dir:
        label = 0

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(directory, filename))
            image = cv2.resize(image, (img_width, img_height))
            image = image / 255.0
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess data from both directories
hamburgers_images, hamburgers_labels = load_and_preprocess_images(hamburgers_dir)
no_hamburgers_images, no_hamburgers_labels = load_and_preprocess_images(no_hamburgers_dir)

# Combine the images and labels for the training data
train_images = np.concatenate((hamburgers_images, no_hamburgers_images))
train_labels = np.concatenate((hamburgers_labels, no_hamburgers_labels))

# Check and balance class distribution if needed
if len(train_images) < 500:
    # If your dataset has fewer than 500 images, consider data augmentation.
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create data generators for training and validation data
    train_datagen = datagen.flow(train_images, train_labels, batch_size=32, seed=42)

# Shuffle the data
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

# Define a pre-trained model for feature extraction
model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"
model = hub.KerasLayer(model_url, trainable=False)

# Define a new classifier
x = layers.Input(shape=(img_height, img_width, 3))
features = model(x)
classifier = layers.Dense(1, activation='sigmoid')(features)  # Changed to 1 output unit

# Create a new model
new_model = Model(inputs=x, outputs=classifier)

# Compile the model with a lower learning rate
new_model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define a checkpoint to save the model
project_folder = "define a checkpoint directory to save the checkpoint"
checkpoint = ModelCheckpoint(project_folder + "model_checkpoint.h5, monitor="val_accuracy", save_best_only=True, mode="max")

# Train the model
epochs = 20
history = new_model.fit(
    train_images,
    train_labels,  # Specify the target labels for hamburgers images
    steps_per_epoch=len(train_images) // 32,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Indicate when the training is complete
print("Model training complete.")

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.legend()
plt.show()

# Save the model to a file
new_model.save("hamburger_model.h5")