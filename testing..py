##use this file to run the model and test against the test set

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Define custom objects (KerasLayer from TensorFlow Hub)
custom_objects = {"KerasLayer": hub.KerasLayer}

# Load the pre-trained model with custom objects
model = load_model('hamburger_model.h5', custom_objects=custom_objects)

# Define image dimensions (should match the dimensions used during training)
img_height, img_width = 279, 181

# Directory containing the new unseen images
test_dir = "pur your directory here"

# Load and preprocess the test images
def load_and_preprocess_test_images(directory):
    images = []
    filenames = []  # Store filenames for debugging
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(directory, filename))
            image = cv2.resize(image, (img_width, img_height))
            image = image / 255.0  # Normalize pixel values
            images.append(image)
            filenames.append(filename)  # Store filenames
    return np.array(images), filenames

test_images, test_filenames = load_and_preprocess_test_images(test_dir)

# Load the labels from the CSV file with a comma separator
labels_df = pd.read_csv('put your file location here', sep=',')

# Debugging: Print the filenames from images and the CSV file
print("Image Filenames:")
print(test_filenames)
print("\nCSV Filenames:")
print(labels_df['filename'].tolist())


# Make predictions on the test images
predictions = model.predict(test_images)

# Display test images and model predictions with filename and labels
for i in range(len(test_images)):
    filename = test_filenames[i]
    contains_hamburger = predictions[i][0] > 0.5
    confidence_percentage = predictions[i][0] * 100

    # Find the corresponding label from the CSV file based on the filename
    label_row = labels_df[labels_df['filename'] == filename]

    if not label_row.empty:
        true_label = "Hamburger" if int(label_row['label'].values[0]) == 1 else "No Hamburger"
    else:
        true_label = "Label not found"

    # Display the image
    plt.imshow(test_images[i])
    plt.axis('off')

    if contains_hamburger:
        plt.title(f'File: {filename}\nTrue Label: {true_label}\nPredicted Label: Hamburger\nConfidence: {confidence_percentage:.2f}%')
    else:
        plt.title(f'File: {filename}\nTrue Label: {true_label}\nPredicted Label: No Hamburger\nConfidence: {100 - confidence_percentage:.2f}%')

    plt.show()

# Extract true labels from the CSV file
true_labels = labels_df['label'].values

# Convert predicted probabilities to binary predictions
predicted_labels = (predictions > 0.5).astype(int)

# Calculate confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Calculate classification metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print and display the metrics and confusion matrix
print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Create a confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Hamburger', 'Hamburger'], rotation=45)
plt.yticks(tick_marks, ['No Hamburger', 'Hamburger'])

thresh = confusion.max() / 2.
for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
    plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center", color="white" if confusion[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Display the classification metrics
plt.figure(figsize=(8, 4))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1], color='skyblue')
plt.title('Classification Metrics')
plt.ylim(0, 1)

# Add actual values on top of the bars
for i, v in enumerate([accuracy, precision, recall, f1]):
    plt.text(i, v + 0.05, f'{v:.2f}', color='black', ha='center', va='bottom')

plt.show()
