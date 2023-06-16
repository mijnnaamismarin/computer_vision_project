import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

from matplotlib import pyplot as plt

from keras.initializers.initializers import RandomNormal
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.python.keras.saving.save import load_model

from utils.image_utils import load_images

print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("GPU Available: ", tf.config.list_physical_devices('GPU'))


class ShadowGAN:

    def __init__(self, shadow_images, object_images, image_size, channels, output_folder):
        self.shadow_images = shadow_images
        self.object_images = object_images
        self.image_size = image_size
        self.channels = channels
        self.image_shape = (self.image_size[0], self.image_size[1], self.channels)
        self.output_folder = output_folder
        self.__latent_dim = 128 * 128

        self.generator = None
        self.discriminator = None
        self.combined = None
        self.__build_models()

    def __build_generator(self):
        model = Sequential()
        model.add(Dense(8 * 8 * 1024, activation="linear", input_dim=self.__latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=512, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=self.channels, kernel_size=[5, 5], strides=[1, 1],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.__latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def __build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), input_shape=self.image_shape,
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=128, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=512, kernel_size=[5, 5], strides=[1, 1],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=1024, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.add(Activation("sigmoid"))

        img = Input(shape=self.image_shape)
        validity = model(img)
        return Model(img, validity)

    def __build_models(self):
        self.optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.optimizer_gen = Adam(learning_rate=0.0002, beta_1=0.5)

        self.discriminator = self.__build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.generator = self.__build_generator()

        shadow_map = Input(shape=(self.__latent_dim,))
        reconstructed_object = self.generator(shadow_map)

        self.discriminator.trainable = False

        validity = self.discriminator(reconstructed_object)

        self.combined = Model(shadow_map, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer_gen)

    def __get_batches(self, shadow_data, object_data, batch_size):
        shadow_batches = []
        object_batches = []

        for i in range(int(shadow_data.shape[0] // batch_size)):
            shadow_batch = shadow_data[i * batch_size:(i + 1) * batch_size]
            object_batch = object_data[i * batch_size:(i + 1) * batch_size]

            normalized_shadow_batch = shadow_batch / 255
            normalized_object_batch = object_batch / 255

            shadow_batches.append(normalized_shadow_batch)
            object_batches.append(normalized_object_batch)

        return shadow_batches, object_batches

    def train(self, epochs, batch_size):
        shadow_arr = self.shadow_images.astype(np.float32)
        object_arr = self.object_images.astype(np.float32)

        valid = np.ones((batch_size, 1))# * random.uniform(0.9, 1.0)
        fake = np.zeros((batch_size, 1))
        cur_epoch = 0
        d_losses = []
        g_losses = []
        for _ in range(epochs):
            cur_epoch += 1
            batch = 0

            for shadows, objects in zip(*self.__get_batches(shadow_arr, object_arr, batch_size)):
                batch += 1

                shadows = shadows.reshape(batch_size, -1)

                generated_object_contours = self.generator.predict(shadows)
                self.discriminator.trainable = True

                d_loss_real = self.discriminator.train_on_batch(objects, valid)
                d_loss_fake = self.discriminator.train_on_batch(generated_object_contours, fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(shadows, valid)

                d_losses.append(d_loss)
                g_losses.append(g_loss)
                print("Current batch " + str(batch) + " in epoch " + str(cur_epoch) + " with D: " + str(
                    d_loss) + " G: " + str(g_loss))

            print("Epoch " + str(cur_epoch))

            plt.plot(d_losses, label='Discriminator', alpha=0.6)
            plt.plot(g_losses, label='Generator', alpha=0.6)
            plt.title("Losses")
            plt.legend()
            plt.savefig(self.output_folder + "/losses_" + str(cur_epoch) + ".png")
            plt.close()

        self.generator.save('generator_model.h5')


    def test(self, image, image_name):
        normalized_image = image

        test_image = normalized_image.reshape(1, -1)

        generated_contours = self.generator.predict(test_image)[0]

        output_image = ((np.squeeze(generated_contours, axis=2)).astype(np.uint8))
        output_image = Image.fromarray(output_image)
        output_image.save(f"{self.output_folder}/{image_name}.png")

    def load_generator(self, model_path):
        self.generator = load_model(model_path, custom_objects={"BatchNormalization": BatchNormalization})
