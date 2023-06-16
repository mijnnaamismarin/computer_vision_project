import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.image_utils import load_images

from gan import ShadowGAN

IMAGE_SIZE = (128, 128)
CHANNELS = 1


def main():
    people_with_masks_dir = "datasets/people_splitted/"
    output_folder = "output/gan"

    shadow_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if
                    '_s_' in file]
    object_masks = [os.path.join(people_with_masks_dir, file) for file in os.listdir(people_with_masks_dir) if
                    '_o_' in file]

    shadow_masks = sorted(shadow_masks)
    object_masks = sorted(object_masks)

    idx_i = 0

    while idx_i < len(object_masks):
        if shadow_masks[idx_i][-5:] != object_masks[idx_i][-5:]:
            shadow_masks.pop(idx_i)
        else:
            idx_i += 1

    shadow_images = load_images(shadow_masks, IMAGE_SIZE)
    object_images = load_images(object_masks, IMAGE_SIZE)

    model = ShadowGAN(shadow_images, object_images, IMAGE_SIZE, CHANNELS, output_folder)

    print(shadow_images.shape)
    print(object_images.shape)

    # Set hyperparameters and train the GAN
    epochs = 100
    batch_size = 64
    model.train(epochs, batch_size)


if __name__ == "__main__":
    main()
