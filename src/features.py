import numpy as np
import src.helpers as helpers
from skimage import io
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os
import glob
import src.dataset as dataset
import skimage.measure as measure
import math as math
import random
from sklearn.preprocessing import StandardScaler


def get_features(filenames, img_height=None, img_width=None, progress=True):
    """
         The function get all features from statistics

         @param filenames:
         @param img_height:
         @param img_width:
         @param progress:
         @return:
            features
    """
    features = []
    if progress:
        helpers.progress(0, len(filenames))
    for i, img_path in enumerate(filenames):
        img = dataset.load_img(img_path, False, False)
        img_features = []
        # img_resize = resize(img, (img_height, img_width), anti_aliasing=True)

        channels = helpers.get_channels(img)

        # mean
        for channel in channels:
            img_features.append(np.mean(channel))

        # var
        for channel in channels:
            img_features.append(np.var(channel))

        # moments_hu
        img_gray = helpers.rgb2gray(img)
        mu = measure.moments_central(img_gray)
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        hu_log = []
        for h in hu:
            hu_log.append(-1 * math.copysign(1.0, h) * math.log10(abs(h)))

        img_features.extend(hu_log)

        # graycomatrix
        # for prop in get_gray_comatrix_features(img):
        #     for p in prop:
        #         img_features.extend(p)

        # #skew = asimetrie
        # for channel in channels:
        #     img_features.append(skew(channel, axis=0, bias=False))
        #
        # #kurtosis = curtoza
        # for channel in channels:
        #     img_features.append(kurtosis(channel, fisher=False))

        features.append(img_features)

        if progress:
            helpers.progress(i, len(filenames), 'Dataset features')

    return np.array(features, dtype=object)


def get_gray_comatrix_features(img):
    """
         The function find features from gray comatrix

         @param filenames:
         @param img_height:
         @param img_width:
         @param progress:
         @return:
            features
    """
    img = helpers.rgb2gray(img)
    bins = list(range(10, 265, 10))
    img_digitized = np.digitize(img, bins)
    gcm = graycomatrix(img_digitized, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=len(bins) + 1)
    # features = ['homogeneity', 'energy', 'correlation']
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    enc = []
    for feat in features:
        enc.append(graycoprops(gcm, feat))
    return enc


def plot_features(features, title, name_of_feature, save=True, show=True):
    """
         The function plot features

         @param features:
         @param title:
         @param name_of_feature:
         @param save:
         @param show:
    """
    fig = plt.figure()
    plt.title(title)

    r = features[:, 0]
    g = features[:, 1]
    b = features[:, 2]

    r.sort()
    g.sort()
    b.sort()

    plt.plot(r, color='r')
    plt.plot(g, color='g')
    plt.plot(b, color='b')
    plt.ylabel(name_of_feature)
    plt.xlabel('Number of image')
    plt.legend(['R channel', 'G channel', 'B channel'], loc="lower left", mode="expand", ncol=3)

    if show:
        plt.show()
    if save:
        fig.savefig('results/' + title + '.jpg')


def get_features_classes(data_train_dir, img_height, img_width):
    """
         The function return features by classes

         @param data_train_dir:
         @param img_height:
         @param img_width:
         @return:
            features
    """
    all_classes_directory = glob.glob(data_train_dir + '/*')
    features = []
    helpers.progress(0, len(all_classes_directory))
    for index, path in enumerate(all_classes_directory):
        class_features = []
        features_classes = get_features(glob.glob(path + '/*'), img_height, img_width, False)

        # mean
        for i in range(0, 3):
            class_features.append(np.mean(features_classes[:, i]))

        # var (https://stats.stackexchange.com/questions/300392/calculate-the-variance-from-variances)
        for i in range(3, 6):
            mean_class = np.mean(features_classes[:, i - 3])
            mean = features_classes[:, i - 3]
            var = features_classes[:, i]
            var_mean = 0
            for j in range(0, len(mean)):
                var_mean = var_mean + ((mean_class - mean[j]) ** 2 + var[j])
            class_features.append(var_mean)

        # # skew = asimetrie
        # for i in range(6, 9):
        #     class_features.append(skew(features_classes[:, i], axis=0, bias=False))
        #
        # # kurtosis = curtoza
        # for i in range(9, 12):
        #     class_features.append(kurtosis(features_classes[:, i], fisher=False))

        features.append(class_features)
        helpers.progress(index, len(all_classes_directory), 'Classes features')

    return np.array(features, dtype=object)


def plot_features_by_classes(features, title, name_of_feature, save=True, show=True):
    """
         The function plot features by classes

         @param features:
         @param title:
         @param name_of_feature:
         @param save:
         @param show:
    """
    r = features[:, 0]
    g = features[:, 1]
    b = features[:, 2]

    fig = plt.figure()

    plt.bar(range(0, np.shape(r)[0]), r, color='r')
    plt.bar(range(0, np.shape(g)[0]), g, color='g')
    plt.bar(range(0, np.shape(b)[0]), b, color='b')

    plt.ylabel(name_of_feature)
    plt.xlabel('Number of image')
    plt.legend(['R channel', 'G channel', 'B channel'], loc="lower left", mode="expand", ncol=3)
    plt.title(title)

    if show:
        plt.show()
    if save:
        fig.savefig('results/' + title + '.jpg')


def rescale_moments_hu(hu):
    """
         The function rescale moments hu

         @param hu:
         @return:
            rescale_hu
    """
    rescale_hu = []
    for h in hu:
        rescale_hu.append(-1 * math.copysign(1.0, h) * math.log10(abs(h)))

    return rescale_hu


def plot_moments_hu(features, title, name_of_feature, save=True, show=True):
    """
         The function plot moments hu

         @param features:
         @param title:
         @param name_of_feature:
         @param save:
         @param show:
    """
    fig = plt.figure()
    plt.title(title)

    hu1 = rescale_moments_hu(features[:, 0])
    hu2 = rescale_moments_hu(features[:, 1])
    hu3 = rescale_moments_hu(features[:, 2])
    hu4 = rescale_moments_hu(features[:, 3])
    hu5 = rescale_moments_hu(features[:, 4])
    hu6 = rescale_moments_hu(features[:, 5])
    hu7 = rescale_moments_hu(features[:, 6])

    hu1.sort()
    hu2.sort()
    hu3.sort()
    hu4.sort()
    hu5.sort()
    hu6.sort()
    hu7.sort()

    plt.plot(hu1, color='red')
    plt.plot(hu2, color='peru')
    plt.plot(hu3, color='olivedrab')
    plt.plot(hu4, color='aqua')
    plt.plot(hu5, color='blueviolet')
    plt.plot(hu6, color='fuchsia')
    plt.plot(hu7, color='navy')

    plt.ylabel(name_of_feature)
    plt.xlabel('Number of image')
    plt.legend(['hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7'], loc="lower left", mode="expand", ncol=7)

    if show:
        plt.show()
    if save:
        fig.savefig('results/' + title + '.jpg')


def get_covariance_matrix(filenames, img_height, img_width, bbox=False):
    """
         The function return covariance matrix

         @param filenames:
         @param img_height:
         @param img_width:
         @param bbox:
         @return:
            covariance
    """
    images = []
    helpers.progress(0, len(filenames))
    for i, img_path in enumerate(filenames):
        img = dataset.load_img(img_path, False, bbox)
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
        # get_covariance_for_img(img_resize)
        images.append(img_resize.reshape(img_resize.shape[0] * img_resize.shape[1] * img_resize.shape[2]))
        helpers.progress(i + 1, len(filenames), 'Covariance matrix images')
    print()

    covariance = [[0 for col in range(len(images))] for row in range(len(images))]
    helpers.progress(0, len(filenames))
    for i, img1 in enumerate(images):
        for j, img2 in enumerate(images):
            covariance[i][j] = get_covariance(img1, img2)
        helpers.progress(i + 1, len(images), 'Covariance matrix')

    fig = plt.figure()
    plt.matshow(covariance)
    plt.show()
    plt.clf()

    return covariance


def get_covariance(x, y):
    """
         The function compute covariance matrix between two random variables

         @param x:
         @param y:
         @return:
            covariance
    """
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar) * (y - ybar)) / (len(x) - 1)


def view_images_with_max_covariance(covariance_matrix, filenames, img_height, img_width):
    """
         The function plot images with max covariance

         @param covariance_matrix:
         @param filenames:
         @param img_height:
         @param img_width:
    """
    for index_row, row in enumerate(covariance_matrix):
        most_covarianted_img_index = np.where(row == np.sort(row)[-2])[0][0]  # except actual image
        most_uncovarianted_img_index = np.argmin(row)

        if index_row in random.sample(range(0, len(filenames)), 50):
            # view img
            img = dataset.load_img(filenames[index_row], False, False)
            img = resize(img, (img_height, img_width), anti_aliasing=True)

            img_covarianted = dataset.load_img(filenames[most_covarianted_img_index], False, False)
            img_covarianted = resize(img_covarianted, (img_height, img_width), anti_aliasing=True)

            img_uncovarianted = dataset.load_img(filenames[most_uncovarianted_img_index], False, False)
            img_uncovarianted = resize(img_uncovarianted, (img_height, img_width), anti_aliasing=True)

            _, axarr = plt.subplots(1, 3, figsize=(9, 3))
            axarr[0].imshow(img)
            axarr[0].set_title('Image')
            axarr[1].imshow(img_covarianted)
            axarr[1].set_title(
                'Biggest positive \n covariance image\n (coef = ' + str(
                    round(row[most_covarianted_img_index], 2)) + ')')
            axarr[2].imshow(img_uncovarianted)
            axarr[2].set_title(
                'Biggest negative \n covariance image\n (coef = ' + str(
                    round(row[most_uncovarianted_img_index], 2)) + ')')
            # plt.show()
            plt.savefig('results/covariance/cov_result_' + str(index_row) + '.jpg')
            plt.clf()


def get_covariances_matrix_for_each_images(filenames, img_height, img_width):
    """
         The function return covariances matrix for each image

         @param filenames:
         @param img_height:
         @param img_width:
         @return:
            covariances
    """
    covariances = []
    helpers.progress(0, len(filenames))
    for i, img_path in enumerate(filenames[13:20]):
        img = dataset.load_img(img_path, False, False)
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
        covv = get_covariance_for_img(img_resize)
        covariances.append(covv)

        # _, axarr = plt.subplots(1, 2, figsize=(9, 6))
        # axarr[0].imshow(img_resize)
        # axarr[0].set_title('Image')
        # pcm = axarr[1].matshow(covv)
        # plt.colorbar(pcm, ax=axarr[1])
        # axarr[1].set_title('covariance matrix (for pixels)')
        # plt.show()

        helpers.progress(i, len(filenames), 'covariance')

    return covariances


def get_covariances_pixel_with_pixel(filenames, img_height, img_width):
    """
         The function return covariance pixel with pixel

         @param filenames:
         @param img_height:
         @param img_width:
         @return:
            covariances
    """
    covariances = []
    helpers.progress(0, len(filenames))
    images = []
    for i, img_path in enumerate(filenames[0:300]):
        img = dataset.load_img(img_path, False, False)
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
        # get_covariance_for_img(img_resize)
        images.append(img_resize)
        helpers.progress(i, len(filenames), 'covariance all pixels')

    print(np.shape(images))
    images = np.array(images)

    for i in range(img_height):
        for j in range(img_width):
            rgb = images[:, i, j, :]
            covariances.append(np.cov(rgb))

    for i in [5000, 5001, 5002, 6000]:
        fig = plt.figure()
        plt.matshow(covariances[i])
        plt.colorbar()
        plt.title('covariance matrix for pixel ' + str(i))
        plt.show()

    return covariances


def get_covariance_for_img(img):
    """
         The function return covariance for image

         @param img:
         @return:
            cov
    """
    pixels = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    cov = np.cov(pixels)

    return cov


def search_similar_image(features, train_filenames, test_filenames, dataset_train_labels, dataset_test_labels):
    """
         The function plot similar images

         @param features:
         @param train_filenames:
         @param test_filenames:
         @param dataset_train_labels:
         @param dataset_test_labels:
    """
    correct = 0
    for index1, filename in enumerate(test_filenames):
        min = 999999999999999999
        argmin = 0
        img = dataset.load_img(filename, False, False)

        enc1 = get_features([filename])

        for index2, enc2 in enumerate(features):
            loss = helpers.mse(enc1, enc2)
            if loss < min:
                argmin = index2
                min = loss

        if dataset_test_labels[index1] == dataset_train_labels[argmin]:
            correct = correct + 1

        img_sim = dataset.load_img(train_filenames[argmin], False, False)

        _, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img)
        axarr[0].set_title('Searched image')
        axarr[1].imshow(img_sim)
        axarr[1].set_title('Most similar image')
        # plt.show()
        plt.savefig('results/search/search_engine_' + str(index1) + '.png')
    print('Accuracy image search: ', correct / len(dataset_test_labels) * 100, '%')

def standardize(features_train, features_test):
    """
        The function standardize features

        @param features_train:
        @param features_test:
        @return
            features_train:
            features_test:
       """
    scale = StandardScaler().fit(features_train)
    features_train = scale.transform(features_train)
    features_test = scale.transform(features_test)

    return features_train, features_test
