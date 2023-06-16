import os
import                                                                   cv2
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt
import tqdm

def split_masks(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    # Get only original images
    orig_images = [file for file in os.listdir(source_dir) if '_' not in file and file.endswith('.jpg')]


    for orig_image in orig_images:
        orig_name = os.path.splitext(orig_image)[0]
        # Copy original image to the destination
        copy2(os.path.join(source_dir, orig_image), destination_dir)

        # Get masks corresponding to the original image
        mask_images = [file for file in os.listdir(source_dir) 
                       if file.startswith(orig_name) and '_' in file]
        # print(mask_images)

        # check if orig_image has been copied to destination
        if not os.path.exists(os.path.join(destination_dir, orig_image)):
            # copy image to destination
            copy2(os.path.join(source_dir, orig_image), destination_dir)


        for mask_image in mask_images:
            img_path = os.path.join(source_dir, mask_image)
            img = cv2.imread(img_path)

            # Flatten the image array and get unique colors
            colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)


            for i, color in enumerate(colors):
                if np.count_nonzero(color) > 0:  # skip black (background)
                    # Create a mask for the current color
                    mask = cv2.inRange(img, color, color)
                    # Make a black image of the same size
                    out = np.zeros_like(img)
                    # Apply the mask
                    out[mask != 0] = color
                    # Save the image
                    out_name = f"{os.path.splitext(mask_image)[0]}_obj_{i}.jpg"
                    cv2.imwrite(os.path.join(destination_dir, out_name), out)


source_folder = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/cars"
destination_folder = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/cars_splitted"

split_masks(source_folder, destination_folder)
