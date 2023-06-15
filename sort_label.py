import os
import shutil
from PIL import Image
from tkinter import Tk, Label, BOTH, YES
from PIL import ImageTk
import cv2
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt

class ImageLabeler:
    def __init__(self, root, dirs, car_dest, people_dest):
        self.root = root
        self.dirs = dirs
        self.car_dest = car_dest
        self.people_dest = people_dest
        self.img_label = Label(root)
        self.img_label.pack(fill=BOTH, expand=YES)
        self.root.bind('<Left>', self.move_car)
        self.root.bind('<Right>', self.move_people)
        self.root.bind('<Down>', self.skip)
        self.images = self.get_images()
        self.current_image = None
        self.current_label = None

        # get the max index of the car_dest and people_dest
        if os.listdir(car_dest):
            self.car_i = max([int(file.split("_")[0][1:]) for file in os.listdir(car_dest)]) + 1
        else:
            self.car_i = 0
        
        if os.listdir(people_dest):
            self.people_i = max([int(file.split("_")[0][1:]) for file in os.listdir(people_dest)]) + 1
        else:
            self.people_i = 0
        self.load_next_image()

    def get_images(self):
        print("Loading images...")
        images = []
        for dir in self.dirs:
            for file in os.listdir(dir):
                if (file.endswith('.jpg')):
                    images.append(os.path.join(dir, file))
        print(f"images: {len(images)}")
        return images

    def load_next_image(self):
        print(f"Images left: {len(self.images)}")
        
        if self.images:
            try:
                self.current_image = self.images.pop(0)
                image = Image.open(self.current_image)
                photo = ImageTk.PhotoImage(image)
                self.img_label.configure(image=photo)
                self.img_label.image = photo
            except FileNotFoundError:
                print(f"Image {self.current_image} not found. Skipping to the next image.")
                self.load_next_image()
        else:
            self.img_label.configure(text="No more images.")

    def split_masks(self, image_name):
        base_name = image_name[:-4]
        source_dir = os.path.dirname(self.current_image)
        destination_dir = self.car_dest if self.current_label == "car" else self.people_dest

        mask_images = [[file, file[-5]] for file in os.listdir(source_dir) if file.endswith('.png') and base_name == file[:-6]]
        mask_images.sort(key=lambda x: x[1])
        print(mask_images)
        
        # get masks_path
        masks_path = [os.path.join(source_dir, mask_image[0]) for mask_image in mask_images]
        # get masks images
        masks_images = [cv2.imread(mask_path) for mask_path in masks_path]
        
        # Flatten the image array and get unique colors
        colors = np.unique(masks_images[0].reshape(-1, masks_images[0].shape[2]), axis=0)

        for i, color in enumerate(colors):
            if np.count_nonzero(color) > 0:  # skip black (background)

                # Create a mask for the current color
                mask_0 = cv2.inRange(masks_images[0], color, color)
                mask_1 = cv2.inRange(masks_images[1], color, color)
                mask_2 = cv2.inRange(masks_images[2], color, color)

                # Make a black image of the same size
                out_0 = np.zeros_like(masks_images[0])
                out_1 = np.zeros_like(masks_images[1])
                out_2 = np.zeros_like(masks_images[2])

                # Apply the mask
                out_0[mask_0 != 0] = color
                out_1[mask_1 != 0] = color
                out_2[mask_2 != 0] = color
                

                # Save the image
                if self.current_label == "car":                  
                    mask_0_name = f"c{str(self.car_i).zfill(4)}_os.png"
                    mask_1_name = f"c{str(self.car_i).zfill(4)}_o.png"
                    mask_2_name = f"c{str(self.car_i).zfill(4)}_s.png"

                    cv2.imwrite(os.path.join(destination_dir, mask_0_name), out_0)
                    cv2.imwrite(os.path.join(destination_dir, mask_1_name), out_1)
                    cv2.imwrite(os.path.join(destination_dir, mask_2_name), out_2)

                    self.car_i += 1
                else:
                    mask_0_name = f"p{str(self.people_i).zfill(4)}_os.png"
                    mask_1_name = f"p{str(self.people_i).zfill(4)}_o.png"
                    mask_2_name = f"p{str(self.people_i).zfill(4)}_s.png"

                    cv2.imwrite(os.path.join(destination_dir, mask_0_name), out_0)
                    cv2.imwrite(os.path.join(destination_dir, mask_1_name), out_1)
                    cv2.imwrite(os.path.join(destination_dir, mask_2_name), out_2)

                    self.people_i += 1

    def move_car(self, event):
        self.current_label = "car"
        self.move_image()
        
    def move_people(self, event):
        self.current_label = "people"
        self.move_image()

    def move_image(self):
        image_name = os.path.basename(self.current_image)
        self.split_masks(image_name)

        # remove the masks from the source directory
        for file in os.listdir(os.path.dirname(self.current_image)):
            if file.endswith('.png') and image_name[:-4] == file[:-6]:
                os.remove(os.path.join(os.path.dirname(self.current_image), file))

        # remove the image from the source directory
        os.remove(self.current_image)


        self.load_next_image()

    def skip(self, event):
        image_name = os.path.basename(self.current_image)

        # remove the masks from the source directory
        for file in os.listdir(os.path.dirname(self.current_image)):
            if file.endswith('.png') and image_name[:-4] == file[:-6]:
                os.remove(os.path.join(os.path.dirname(self.current_image), file))
                
        # remove the image from the source directory
        os.remove(self.current_image)

        self.load_next_image()

if __name__ == "__main__":
    root = Tk()

    # you need to extract the SOBA_v2.zip file in the data directory
    base_dir = 'SOBA_v2/SOBA/SOBA/'

    # the different subdirectories which we will need
    image_dirs = [base_dir + 'ADE/', base_dir + 'challenge/', base_dir + 'COCO/', base_dir + 'SBU/', base_dir + 'SBU-test/', base_dir + 'WEB/']

    # the destination directories for the sorted images
    car_dest = 'data/processed_data/cars'
    people_dest = 'data/processed_data/people'

    # if the locations do not exist, create them
    if not os.path.exists(car_dest):
        os.makedirs(car_dest)
    if not os.path.exists(people_dest):
        os.makedirs(people_dest)

    ImageLabeler(root, image_dirs, car_dest, people_dest)
    root.mainloop()