import os
import shutil
from PIL import Image
from tkinter import Tk, Label, BOTH, YES
from PIL import ImageTk

class ImageClassifier:
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
        self.load_next_image()

    def get_images(self):
        print("Loading images...")
        print(self.dirs)
        images = []
        for dir in self.dirs:
            print(dir)
            for file in os.listdir(dir):
                print(file)

                # check if file has three - symbols in it
                b = file.split('-')
        
                if (file.endswith('.jpg') or file.endswith('.png')) and ('-' not in file[-6:] or len(b) <= 3):
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

    def move_car(self, event):
        self.move_image(self.car_dest)
        
    def move_people(self, event):
        self.move_image(self.people_dest)

    def move_image(self, dest):
        base = os.path.basename(self.current_image)
        prefix = os.path.splitext(base)[0]
        dir = os.path.dirname(self.current_image)
        for file in os.listdir(dir):
            if file.startswith(prefix):
                shutil.move(os.path.join(dir, file), dest)
        self.load_next_image()

    def skip(self, event):
        # remove the image
        base = os.path.basename(self.current_image)
        prefix = os.path.splitext(base)[0]
        dir = os.path.dirname(self.current_image)
        for file in os.listdir(dir):
            if file.startswith(prefix):
                os.remove(os.path.join(dir, file))

        self.load_next_image()

root = Tk()
base_dir = '/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/SOBA/'
# image_dirs = [base_dir + 'ADE/', base_dir + 'challenge/', base_dir + 'COCO/', base_dir + 'SBU/', base_dir + 'SBU-test/', base_dir + 'WEB/']
image_dirs = [base_dir + 'challenge/', base_dir + 'SBU-test/', base_dir + 'WEB/']
car_dest = '/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/cars'
people_dest = '/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/people'
ImageClassifier(root, image_dirs, car_dest, people_dest)
root.mainloop()
