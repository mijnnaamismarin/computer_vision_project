import os

from utils.image_utils import load_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gan import ShadowGAN

IMAGE_SIZE = (128, 128)
CHANNELS = 1


def main():
    test_people_with_masks_dir = "datasets/people_testset"
    output_folder = "output/gan"

    test_shadow_masks = [os.path.join(test_people_with_masks_dir, file) for file in
                         os.listdir(test_people_with_masks_dir) if
                         '_s_' in file]

    test_shadow_masks = sorted(test_shadow_masks)

    test_shadow_images = load_images(test_shadow_masks, IMAGE_SIZE)

    model = ShadowGAN([], [], IMAGE_SIZE, CHANNELS, output_folder)
    model.load_generator('generator_model.h5')

    test_image1_name = test_shadow_masks[0].rsplit("/", 1)[-1]
    test_image1 = test_shadow_images[0]

    model.test(test_image1, test_image1_name)


if __name__ == "__main__":
    main()
