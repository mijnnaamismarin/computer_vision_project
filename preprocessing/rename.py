import os
from shutil import move


def rename_files_and_masks_in_dir(path):
    i = 1
    mask_conversion = {"-1": "_os", "-2": "_o", "-3": "_s"}

    # Sort the files so that original images and their masks are grouped together
    files = os.listdir(path)
    files.sort()
    # loop over the length of files in steps of 4
    r = range(0, len(files), 4)
    for j in r:
        # Generate new name base (incremented number), padded with zeroes
        # new_name_base = "p" + str(i).zfill(4) #for people
        new_name_base = "c" + str(i).zfill(4)  # for cars

        # select the first three files which are the masks
        mask_os = files[j]
        mask_o = files[j + 1]
        mask_s = files[j + 2]
        # select the last file which is the original image
        orig = files[j + 3]
        # rename original to new_name_base
        os.rename(os.path.join(path, orig), os.path.join(path, new_name_base + ".jpg"))
        # rename masks to new_name_base + mask_conversion
        os.rename(os.path.join(path, mask_os), os.path.join(path, new_name_base + "_os.jpg"))
        os.rename(os.path.join(path, mask_o), os.path.join(path, new_name_base + "_o.jpg"))
        os.rename(os.path.join(path, mask_s), os.path.join(path, new_name_base + "_s.jpg"))
        i += 1

    # for filename in files:
    #     if filename.endswith(('.jpg', '.png')):
    #         # Generate new name base (incremented number), padded with zeroes
    #         new_name_base = "p" + str(i).zfill(4) #for people
    #         # new_name_base = "c" + str(i).zfill(4) #for cars

    #         # Detect if it's a mask or not
    #         is_mask = any(key in filename for key in mask_conversion)

    #         if not is_mask:  # if it's not a mask, we increment the base name for the next files
    #             i += 1

    #         new_name = new_name_base
    #         for key in mask_conversion:
    #             if key in filename:
    #                 new_name += mask_conversion[key]
    #                 break

    #         # Preserve the original file extension
    #         _, ext = os.path.splitext(filename)
    #         new_name += ext

    #         src = os.path.join(path, filename)
    #         dst = os.path.join(path, new_name)
    #         move(src, dst)
    #         print(f"Renamed file {src} to {dst}")


path_to_destination_folder = "/home/marin/Documents/University/CV/project/SOBA_v2/SOBA/processed/cars"
rename_files_and_masks_in_dir(path_to_destination_folder)
