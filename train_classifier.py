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
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
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

else: # This entire else statement is not needed if you load the data properly. It just preprocesses data which comes from the sort_label.py
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
    early_stopping = EarlyStopping(monitor='accuracy', patience=10)

    # Create a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout layer after the first Conv2D layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout layer after the second Conv2D layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout layer after the third Conv2D layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer after the Dense layer
    model.add(Dense(1, activation='sigmoid'))


    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    # Create an instance of RandomOverSampler
    ros = RandomOverSampler()

    # Reshape the input data for oversampling
    X_train_reshaped = X_train_processed.reshape(X_train_processed.shape[0], -1)

    # Perform random oversampling on the reshaped data
    X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_reshaped, y_train_processed)

    # Reshape the oversampled data back to the original image shape
    X_train_oversampled = X_train_oversampled.reshape(X_train_oversampled.shape[0], *input_shape)

    # Create an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.025,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.025,
        horizontal_flip=True,
        fill_mode="nearest")

    # Fit the ImageDataGenerator on the augmented data
    datagen.fit(X_train_oversampled)

    # Create a new list to store the augmented samples
    augmented_images = []

    # Generate augmented samples using the ImageDataGenerator
    for image in X_train_oversampled:
        batch = np.expand_dims(image, axis=0)
        aug_iter = datagen.flow(batch, batch_size=1)
        augmented_image = next(aug_iter)[0]
        augmented_images.append(augmented_image)

    # show 3 augmented images and 3 original images
    for i in range(3):
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train_oversampled[i])
        plt.axis("off")
        plt.subplot(3, 3, i + 4)
        plt.imshow(augmented_images[i])
        plt.axis("off")
    plt.show()




    # Convert the augmented images to a numpy array
    X_train_augmented = np.array(augmented_images)

    # Concatenate the augmented samples with the original data
    X_train_final = np.concatenate((X_train_processed, X_train_augmented))
    y_train_final = np.concatenate((y_train_processed, y_train_oversampled))

    # Shuffle the data
    indices = np.random.permutation(len(X_train_final))
    X_train_final = X_train_final[indices]
    y_train_final = y_train_final[indices]
    
    # # Perform data augmentation and oversampling for the underrepresented class
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest")
    
    # model.fit(X_train_final, y_train_final, epochs=200, callbacks=[early_stopping], validation_data=(X_test_processed, y_test_processed))
    # Fit the model with augmented and oversampled data
    model.fit_generator(
        datagen.flow(X_train_final, y_train_final, batch_size=32),
        steps_per_epoch=len(X_train_oversampled) // 32,
        epochs=200,
        callbacks=[early_stopping],
        validation_data=(X_test_processed, y_test_processed)
    )

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
