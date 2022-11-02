from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import os
import glob
import numpy as np


# -----------------------------------------------------------------------------
# Load dataset paths
# -----------------------------------------------------------------------------
def load_dataset_paths(data_train_dir, data_test_dir):
    dataset_train = load_files(
        data_train_dir,
        shuffle=False,
        load_content=False
    )

    dataset_test = load_files(
        data_test_dir,
        shuffle=False,
        load_content=False
    )

    class_names = dataset_train.target_names
    print('Class number: ', len(class_names))
    print('Classes: ')
    print([cls.split('-')[-1] for cls in class_names])
    print('Train images: ', len(dataset_train.filenames))
    print('Test images: ', len(dataset_test.filenames))

    return dataset_train, dataset_test


# -----------------------------------------------------------------------------
# Example of images from dataset group by classes
# -----------------------------------------------------------------------------
def dataset_examples_each_class(data_train_dir, img_height, img_width, show=True):
    for interval in [[0, 60], [60, 120]]:
        fig = plt.figure(figsize=(18, 9))
        for index, class_dir in enumerate(glob.glob(data_train_dir + '/*')[interval[0]: interval[1]]):
            plt.subplot(6, 10, index + 1)
            img = io.imread(glob.glob(class_dir + '/*')[4])
            img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
            plt.imshow(img_resize, cmap='gray')
            plt.title((class_dir.split('\\')[-1]).split('-')[-1])

        plt.tight_layout()
        if show:
            plt.show()
        fig.savefig('results/training_data_visualisation_' + str(interval[0]) + '-' + str(interval[1]) + '.jpg')


# -----------------------------------------------------------------------------
# Dataset distribution
# -----------------------------------------------------------------------------
def dataset_distribution(data_train_dir):
    all_classes_directory = glob.glob(data_train_dir + '/*')
    class_images_distribution = []
    class_labels = []
    for path in all_classes_directory:
        class_images_distribution.append(len(glob.glob(path + '/*')))
        class_labels.append((path.split('\\')[-1]).split('-')[-1])

    x_pos = [i for i, _ in enumerate(class_labels)]

    fig, ax = plt.subplots(figsize=(18, 9))
    plt.bar(x_pos, class_images_distribution, color='green')
    plt.xlabel("Number of images", fontsize=12)
    plt.ylabel("Class", fontsize=12)
    plt.title("Data distribution", fontsize=16)
    plt.xticks(x_pos, class_labels, fontsize=8, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    fig.savefig('results/training_data_distribution.jpg')
