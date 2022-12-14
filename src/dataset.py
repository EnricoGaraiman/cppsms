from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import os
import glob
import numpy as np
from src.helpers import rgb2gray
import src.helpers as helpers

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
            # img = io.imread(glob.glob(class_dir + '/*')[4])
            img = load_img(glob.glob(class_dir + '/*')[4], False, True)
            # img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
            plt.imshow(img, cmap='gray')
            plt.title((class_dir.split('\\')[-1]).split('-')[-1])

        plt.tight_layout()
        if show:
            plt.show()
        # fig.savefig('results/training_data_visualisation_' + str(interval[0]) + '-' + str(interval[1]) + '.jpg')


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
    # fig.savefig('results/training_data_distribution.jpg')


# -----------------------------------------------------------------------------
# Load Image
# -----------------------------------------------------------------------------
def load_img(filename, grayscale, bbox=None, img_height=False, img_width=False):
    img = plt.imread(filename)

    # if img.max() < 2: img = np.uint8(255 * img)
    if grayscale: img = rgb2gray(img)
    if bbox:
        text = open('stanfordDogsDataset/Annotation/' + filename.split('\\', 1)[-1].replace('.jpg', '').replace('\\',
                                                                                                                '/')).read()
        x_min = int(text.split('<xmin>')[1].split('</xmin>')[0])
        x_max = int(text.split('<xmax>')[1].split('</xmax>')[0])
        y_min = int(text.split('<ymin>')[1].split('</ymin>')[0])
        y_max = int(text.split('<ymax>')[1].split('</ymax>')[0])
        img = img[y_min:y_max, x_min:x_max]

    if img_width and img_height:
        img = resize(img, (img_height, img_width), anti_aliasing=True)

    return img


# -----------------------------------------------------------------------------
# DISPLAY IMG BY INDEX
# -----------------------------------------------------------------------------
def display_img_by_index(data, index, features, title, save=True, show=True):
    fig = plt.figure()

    plt.title(title + ' (Value: ' + str(features[index]) + ')')
    io.imshow(data[index])

    if show:
        plt.show()
    if save:
        fig.savefig('results/' + title + '.jpg')


# -----------------------------------------------------------------------------
# DISPLAY LOAD DATASET
# -----------------------------------------------------------------------------
def load_dataset(filenames, bbox=None, grayscale=False, img_height=False, img_width=False):
    images = []

    helpers.progress(0, len(filenames))
    for i, filename in enumerate(filenames):
        img = load_img(filename, grayscale, bbox, img_height, img_width)
        images.append(img)
        helpers.progress(i, len(filenames), 'Dataset images')

    return images
