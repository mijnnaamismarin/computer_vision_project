import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


def preprocess_image(image_path, input_shape):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert image to grayscale
    # Resize the image to the desired input shape
    img = img.resize((input_shape[0], input_shape[1]))  # Double the size
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize pixel values between 0 and 1
    img_array = img_array / 255.0
    # Return the preprocessed image data
    return img_array


# Define paths for the people with masks and cars with masks datasets
people_with_masks_dir = "people_splitted"
cars_with_masks_dir = "cars_splitted"

# Define the local root directory to store preprocessed images
local_root = "/home/marin/Documents/University/CV/project/data_selection/data/"
# data_root = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/"

# # Create the local directories if they don't exist
# local_people_dir = os.path.join(local_root, "people")
# local_cars_dir = os.path.join(local_root, "cars")
# os.makedirs(local_people_dir, exist_ok=True)
# os.makedirs(local_cars_dir, exist_ok=True)

# Load all the people with masks (_s)
# if file contains _s_
people_with_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if
                     '_s_' in file]

# Load all the cars with masks (_c)
cars_with_masks = [os.path.join(cars_with_masks_dir, file) for file in os.listdir(cars_with_masks_dir) if '_s_' in file]

# Create labels for the datasets (_s: 0, _c: 1)
labels = [0] * len(people_with_masks) + [1] * len(cars_with_masks)

# Combine the datasets and labels
data = people_with_masks + cars_with_masks

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define image dimensions
input_shape = (256, 256, 1)  # Adjust the dimensions based on your dataset and make sure it's grayscale

# Preprocess and load the training data
X_train_processed = []
for image_path in X_train:
    filename = os.path.basename(image_path)
    local_path = os.path.join(local_root, filename)
    preprocessed_image = preprocess_image(image_path, input_shape)
    Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(local_path)
    preprocessed_image = preprocessed_image.reshape(*preprocessed_image.shape, 1)
    X_train_processed.append(preprocessed_image)

X_train_processed = np.array(X_train_processed)
y_train_processed = np.array(y_train)

# Preprocess and load the test data
X_test_processed = []
for image_path in X_test:
    filename = os.path.basename(image_path)
    local_path = os.path.join(local_root, filename)
    preprocessed_image = preprocess_image(image_path, input_shape)
    Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(local_path)
    preprocessed_image = preprocessed_image.reshape(*preprocessed_image.shape, 1)
    X_test_processed.append(preprocessed_image)
X_test_processed = np.array(X_test_processed)
y_test_processed = np.array(y_test)

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

    early_stopping = EarlyStopping(monitor='accuracy', patience=3)

    # Create a CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    # model.fit(X_train_processed, y_train_processed, batch_size=32, epochs=25, callbacks=[early_stopping])   
    model.fit(datagen.flow(X_train_processed, y_train_processed, batch_size=32),
              validation_data=(X_test_processed, y_test_processed),
              steps_per_epoch=len(X_train_processed) // 32, epochs=25, callbacks=[early_stopping])
    # Save the trained model
    model.save(model_path)

# Evaluate the model on the test set
y_pred = model.predict(X_test_processed)

# Convert the predictions to binary values <0.5 is 0 and >=0.5 is 1>
y_pred = np.where(y_pred < 0.5, 0, 1)

accuracy = accuracy_score(y_test_processed, y_pred)
print("Accuracy:", accuracy)
