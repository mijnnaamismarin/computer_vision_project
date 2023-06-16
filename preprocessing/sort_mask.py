import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
from shutil import copy2
import numpy as np


class MaskSelector:
    def __init__(self, source_folder, destination_folder):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.image_files = sorted([f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png'))])

        # Create destination directory if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Keyboard keys mapping
        self.keys_map = {81: 'left', 82: 'down', 83: 'right', 84: 'up'}

        # Start the selection process
        self.select_masks()

    def select_masks(self):
        index = 0
        current_original = None
        while index < len(self.image_files):
            file_name = self.image_files[index]

            if "-os" in file_name:
                mask = cv2.imread(os.path.join(self.source_folder, file_name))

                cv2.imshow('Image - press left to keep, right to discard: ' + file_name, mask)
                k = cv2.waitKey(0)
                if k in self.keys_map.keys():
                    if self.keys_map[k] == 'left':  # keep mask
                        # Copy all versions of masks (-os, -s, -o)
                        base_name = file_name.split('-os')[0]
                        suffix_name = file_name.split('-os')[1]

                        # Copy the original if not yet copied
                        if not os.path.exists(os.path.join(self.destination_folder, base_name + ".jpg")):
                            copy2(os.path.join(self.source_folder, base_name + ".jpg"), self.destination_folder)

                        for suffix in ['-os', '-s', '-o']:
                            mask_name = base_name + suffix + suffix_name
                            if mask_name in self.image_files:
                                copy2(os.path.join(self.source_folder, mask_name), self.destination_folder)
                    elif self.keys_map[k] == 'right':  # discard mask
                        pass
                cv2.destroyAllWindows()
            else:
                current_original = file_name

            index += 1
        cv2.destroyAllWindows()


source_folder = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/people_splitted"
destination_folder = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/people_selected"

selector = MaskSelector(source_folder, destination_folder)
