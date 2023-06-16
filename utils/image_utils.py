from flatbuffers.builder import np
from keras.utils import load_img, img_to_array
from PIL import Image

def load_images(image_paths, image_size):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_size, color_mode="grayscale")
        img = img_to_array(img).astype('float32')
        threshold = 10
        img[img >= threshold] = 255
        img[img < threshold] = 0
        images.append(img)
    return np.array(images)
