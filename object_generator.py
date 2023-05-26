import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
from keras import initializers
import numpy as np

from keras.utils import img_to_array, load_img
from tqdm import tqdm

print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("GPU Available: ", tf.test.is_gpu_available())

# Load the images
def load_images(image_paths, image_size):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_size, color_mode="grayscale")
        img = img_to_array(img).astype('float32')
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)

# Define image size
image_size = (64, 64)

# Define paths for the people with masks and cars with masks datasets
people_with_masks_dir = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/people_splitted"

# Define the local root directory to store preprocessed images
local_root = "/home/marin/Documents/University/CV/project/data_selection/data_gan/"

shadow_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if '_s_' in file]
object_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if '_o_' in file]

# Load and preprocess images
shadow_images = load_images(shadow_masks, image_size)
object_images = load_images(object_masks, image_size)


# PROPERLY PREPROCESS THE DATA 

# Let's assume that your shadows and objects are 64x64 grayscale images
img_rows = 64
img_cols = 64
channels = 1
img_shape = (img_rows, img_cols, channels)



# Generator
def build_generator():
    noise_shape = (100,)
    
    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

# Discriminator
def build_discriminator():
    img_shape = (img_rows, img_cols, channels)

    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Building and compiling the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(0.0002, 0.5), 
                      metrics=['accuracy'])

# Build and compile the generator
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=Adam())

# The generator takes noise as input and generated imgs
z = Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam())

# Number of epochs and batch size
epochs = 30000
batch_size = 128

# Training loop
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs)):
    # Train Discriminator
    idx = np.random.randint(0, object_images.shape[0], batch_size)
    real_imgs = object_images[idx]
    
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generate a batch of new images
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    g_loss = combined.train_on_batch(noise, real)

    # If at save interval => save generated image samples and models
    if epoch % 2000 == 0:
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

# Save final models
generator.save('gan_generator.h5')
discriminator.save('gan_discriminator.h5')