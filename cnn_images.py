

# Import necessary libraries for data manipulation, machine learning, and image processing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
# Mount Google Drive to access files stored there
drive.mount('/content/drive')

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to training, validation, and test label files and image directories
#It should be noted that in our drive there is a
#folder called data and from there you get to the relevant folders and files.
train_labels_path = '/content/drive/MyDrive/data/train-c3-labels.xlsx'
val_labels_path = '/content/drive/MyDrive/data/val-c3-labels.xlsx'
train_path = '/content/drive/MyDrive/data/train-C3'
val_path = '/content/drive/MyDrive/data/val-C3'
test_path = '/content/drive/MyDrive/data/test-C3'

# Load training and validation labels from Excel files
train_labels = pd.read_excel(train_labels_path)
val_labels = pd.read_excel(val_labels_path)

# Convert 'id' values to strings and append '.tiff' extension for image file names
train_labels['id'] = train_labels['id'].astype(str) + '.tiff'
val_labels['id'] = val_labels['id'].astype(str) + '.tiff'

# Convert 'label' values to strings
train_labels['label'] = train_labels['label'].astype(str)
val_labels['label'] = val_labels['label'].astype(str)

# Check the file names in the training directory
print("Training images:")
print(os.listdir(train_path)[:5])  # Display the first 5 file names

# Check the file names in the validation directory
print("Validation images:")
print(os.listdir(val_path)[:5])  # Display the first 5 file names

# Create data generators for training and validation with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generator for training data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory=train_path,
    x_col='id',
    y_col='label',
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

# Create data generator for validation data
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_labels,
    directory=val_path,
    x_col='id',
    y_col='label',
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

# Define a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Create a data generator for test data with rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a DataFrame with file names for the test data
test_filenames = [filename for filename in os.listdir(test_path) if filename.endswith('.tiff')]
test_df = pd.DataFrame({'id': test_filenames})

# Create a data generator for test data
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_path,
    x_col='id',
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# Predict results on the test data
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)

# Save the results to a CSV file
# Use split('/')[-1] to get the filename without the path and split('.')[0] to remove the file extension
image_ids = [filename.split('/')[-1].split('.')[0] for filename in test_generator.filenames]
# Create a DataFrame with image IDs and predicted labels, then save it to a CSV file on Google Drive
results = pd.DataFrame({'id': image_ids, 'label': predicted_labels.flatten()})
results.to_csv('/content/drive/MyDrive/data/submission.csv', index=False)

print("Results have been saved to submission.csv")