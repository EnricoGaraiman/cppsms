from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import os
import glob
import numpy as np
from src.helpers import rgb2gray
import src.helpers as helpers
from numpy import save, load
# from rembg import remove
import cv2


# -----------------------------------------------------------------------------
# Load dataset paths
# -----------------------------------------------------------------------------
def load_dataset_paths(data_train_dir, data_test_dir):
    """
           The function receives the dataset dirs and dataset and classes.

           @param data_train_dir:
           @param data_test_dir:
           @return:
               dataset_train
               dataset_test
               classes
    """
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
    classes = [cls.split('-')[-1] for cls in class_names]
    print('Class number: ', len(class_names))
    print('Classes: ')
    print(classes)
    print('Train images: ', len(dataset_train.filenames))
    print('Test images: ', len(dataset_test.filenames))

    return dataset_train, dataset_test, classes


# -----------------------------------------------------------------------------
# Example of images from dataset group by classes
# -----------------------------------------------------------------------------
def dataset_examples_each_class(data_train_dir, img_height, img_width, show=True):
    """
         The function receives the dataset dirs and display examples from each class

         @param data_train_dir:
         @param img_height:
         @param img_width:
         @param show:
    """
    for interval in [[0, 60], [60, 120]]:
        fig = plt.figure(figsize=(18, 9))
        for index, class_dir in enumerate(glob.glob(data_train_dir + '/*')[interval[0]: interval[1]]):
            plt.subplot(6, 10, index + 1)
            # img = io.imread(glob.glob(class_dir + '/*')[4])
            img = load_img(glob.glob(class_dir + '/*')[4], False, False)
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
    """
         The function display dataset distribution

         @param data_train_dir:
    """
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
def load_img(filename, grayscale, bbox=None, img_height=False, img_width=False, remove_bg=False, type=''):
    """
         The function receives filename and sizes and return image as array

         @param filename:
         @param grayscale:
         @param bbox:
         @param img_height:
         @param img_width:
         @param remove_bg:
         @param type:
         @return:
            img
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    # if remove_bg:
    #     img = remove(img)
    #     if not os.path.isdir(r'stanfordDogsDataset/split_images_crop_no_bg/' + type + '/' + (filename.split('\\', 1)[-1].replace('.jpg', '').replace('\\', '/')).rsplit('/', 1)[0]):
    #         os.mkdir(r'stanfordDogsDataset/split_images_crop_no_bg/' + type + '/' + (filename.split('\\', 1)[-1].replace('.jpg', '').replace('\\', '/')).rsplit('/', 1)[0])
    #
    #     cv2.imwrite(r'stanfordDogsDataset/split_images_crop_no_bg/' + type + '/' + (filename.split('\\', 1)[-1].replace('.jpg', '').replace('\\', '/')).rsplit('/', 1)[0] + '/' + filename.rsplit('\\', 1)[-1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if img_width and img_height:
        img = resize(img, (img_height, img_width)) #, anti_aliasing=True)

    return img


# -----------------------------------------------------------------------------
# DISPLAY IMG BY INDEX
# -----------------------------------------------------------------------------
def display_img_by_index(data, index, features, title, save=True, show=True):
    """
         The function receive a image and display

         @param data:
         @param index:
         @param features:
         @param title:
         @param save:
         @param show:
    """
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
def load_dataset(filenames, bbox=None, grayscale=False, img_height=False, img_width=False, remove_bg=False, type=''):
    """
         The function load dataset

         @param filenames:
         @param bbox:
         @param grayscale:
         @param img_height:
         @param img_width:
         @param remove_bg:
         @param type:
         @return:
            images
    """
    images = []

    helpers.progress(0, len(filenames))
    for i, filename in enumerate(filenames):
        img = load_img(filename, grayscale, bbox, img_height, img_width, remove_bg, type)
        images.append(img)
        helpers.progress(i, len(filenames), 'Dataset images')

    return images


def save_features(features, featuresName):
    """
         The function save features on disk

         @param features:
         @param featuresName:
    """
    for i, feature in enumerate(features):
        save('results/features/' + featuresName[i] + '.npy', feature)


# def load_features_train():
#     tfidf_train = load('results/features/tfidf_train.npy')
#     descriptors_train = load('results/features/descriptors_train.npy', allow_pickle=True)
#     frequency_vectors_train = load('results/features/frequency_vectors_train.npy')
#     dataset_train_labels = load('results/features/dataset_train_labels.npy')
#     features_train = [] #load('results/features/features_train.npy')
#
#     return tfidf_train, descriptors_train, frequency_vectors_train, dataset_train_labels, features_train
#
#
# def load_features_test():
#     tfidf_test = load('results/features/tfidf_test.npy')
#     descriptors_test = load('results/features/descriptors_test.npy', allow_pickle=True)
#     frequency_vectors_test = load('results/features/frequency_vectors_test.npy')
#     dataset_test_labels = load('results/features/dataset_test_labels.npy')
#     features_test = [] #load('results/features/features_test.npy')
#
#     return tfidf_test, descriptors_test, frequency_vectors_test, dataset_test_labels, features_test
