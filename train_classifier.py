import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from PIL import Image
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras.applications import VGG16


def preprocess_image(image_path, input_shape, threshold=128):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert image to grayscale

    # Apply binary thresholding
    img = img.point(lambda p: p > threshold and 255)

    # Resize the image to the desired input shape
    img = img.resize((input_shape[0], input_shape[1]))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Normalize pixel values between 0 and 1
    img_array = img_array / 255.0

    # If more than 99.9% of the image is black then return None
    if np.mean(img_array) < 0.001:
        return None

    # Stack the grayscale image 3 times to create a 3-channel image
    img_array = np.stack([img_array] * 3, axis=-1)

    # Return the preprocessed image data
    return img_array


# Define paths for the people with masks and cars with masks datasets
people_with_masks_dir = "data/processed_data/people"
cars_with_masks_dir = "data/processed_data/cars"

# Define the local root directory to store preprocessed images
local_root = "data/"

# Define the local root directories to store preprocessed images
local_train_root = "data/train_preprocessed"
local_test_root = "data/test_preprocessed"

# Define image dimensions
input_shape = (256, 256, 3)  # Adjust the dimensions based on your dataset and make sure it's grayscale

# Check if the preprocessed directories already exist
if os.path.exists(local_train_root) and os.path.exists(local_test_root):
    # If preprocessed directories exist, load the images directly
    X_train_processed = [np.array(Image.open(img_path)) for img_path in glob.glob(f"{local_train_root}/*.png")]
    X_test_processed = [np.array(Image.open(img_path)) for img_path in glob.glob(f"{local_test_root}/*.png")]

    X_train_processed = np.array(X_train_processed)
    X_test_processed = np.array(X_test_processed)

    X_train_processed = X_train_processed.reshape(*X_train_processed.shape, 1)
    X_test_processed = X_test_processed.reshape(*X_test_processed.shape, 1)

    X_train_processed = np.repeat(X_train_processed, 3, axis=-1)
    X_test_processed = np.repeat(X_test_processed, 3, axis=-1)


    # Assume the labels are in the filenames in the form of "c*" and "p*"
    y_train_processed = [os.path.basename(img_path)[0] for img_path in glob.glob(f"{local_train_root}/*.png")]
    y_test_processed = [os.path.basename(img_path)[0] for img_path in glob.glob(f"{local_test_root}/*.png")]

    # convert "c" to 1 and "p" to 0
    y_train_processed = np.array(y_train_processed)
    y_test_processed = np.array(y_test_processed)
    y_train_processed = np.where(y_train_processed == "c", 1, 0)
    y_test_processed = np.where(y_test_processed == "c", 1, 0)

else:
    # Create directories if they do not exist
    os.makedirs(local_train_root, exist_ok=True)
    os.makedirs(local_test_root, exist_ok=True)

    # Load all the people with masks (_s)
    # if file contains _s_
    people_with_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if '_s.' in file]

    # Load all the cars with masks (_c)
    cars_with_masks = [os.path.join(cars_with_masks_dir, file) for file in os.listdir(cars_with_masks_dir) if '_s.' in file]

    # Create labels for the datasets (_s: 0, _c: 1)
    labels = [0] * len(people_with_masks) + [1] * len(cars_with_masks)

    # Combine the datasets and labels
    data = people_with_masks + cars_with_masks

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Preprocess and load the training data
    X_train_processed = []
    y_train_processed = []  # New list for processed labels
    for image_path, label in zip(X_train, y_train):
        preprocessed_image = preprocess_image(image_path, input_shape)
        if preprocessed_image is None:
            continue
        filename = os.path.basename(image_path)
        local_path = os.path.join(local_train_root, filename)
        Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(local_path)
        X_train_processed.append(preprocessed_image)
        y_train_processed.append(label)  # Append label only if image is not None

    X_train_processed = np.array(X_train_processed)
    y_train_processed = np.array(y_train_processed)

    # Preprocess and load the test data
    X_test_processed = []
    y_test_processed = []  # New list for processed labels
    for image_path, label in zip(X_test, y_test):
        preprocessed_image = preprocess_image(image_path, input_shape)
        if preprocessed_image is None:
            continue
        filename = os.path.basename(image_path)
        local_path = os.path.join(local_test_root, filename)
        Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(local_path)
        X_test_processed.append(preprocessed_image)
        y_test_processed.append(label)  # Append label only if image is not None

    X_test_processed = np.array(X_test_processed)
    y_test_processed = np.array(y_test_processed)


# Define the path to save the model
model_path = "path_to_save_model"
force_train = True

if os.path.exists(model_path) and not force_train:
    # Load the saved model
    model = load_model(model_path)
else:

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    datagen.fit(X_train_processed)

    early_stopping = EarlyStopping(monitor='accuracy', patience=2)

    # Create a modified VGG16 model for grayscale images
    baseModel = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    inputs = Input(shape=input_shape)
    x = baseModel(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(datagen.flow(X_train_processed, y_train_processed, batch_size=32),
              validation_data=(X_test_processed, y_test_processed),
              steps_per_epoch=len(X_train_processed) // 32, epochs=25, callbacks=[early_stopping])

    # Save the trained model
    model.save(model_path)

# Evaluate the model on the test set
y_pred = model.predict(X_test_processed)

# Convert the predictions to binary values <0.5 is 0 and >=0.5 is 1>
y_pred = np.where(y_pred < 0.5, 0, 1)

# Print classification report
print(classification_report(y_test_processed, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_processed, y_pred)

# Print the confusion matrix
print('Confusion Matrix')
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the confusion matrix using Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
