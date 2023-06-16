This repository contains the project materials for the Computer Vision Seminar course at TU Delft. The primary goal of this project was to establish a machine learning pipeline that can accurately classify shadows of people and cars, based on the manual preprocessing of the SOBA dataset.
Repository Structure

The repository consists of the following python files:

    sort_label.py - Python script for manual preprocessing of the SOBA_v2 dataset.
    train_classifier.py - Python script for loading in the dataset and training a machine learning model.

Dataset

The dataset used for this project is the SOBA_classification dataset, which can be accessed through the following link.

To use this dataset with the scripts in this repository, follow these steps:

    Download the SOBA_classification dataset from the provided Google Drive.
    Extract the dataset in the root directory of this project.
    Update the dataset paths in both the train_classifier.py and training.py scripts.
    
If you want to preprocess the data again through the sort_label.py you follow the same steps but now download the SOBA_v2 dataset from the Google Drive
