import numpy as np
import src.helpers as helpers
from skimage import io
from skimage.transform import resize
from scipy.stats import skew,kurtosis
import matplotlib.pyplot as plt
import os
import glob


def get_features(filenames, img_height, img_width, progress=True):
    features = []
    if progress:
        helpers.progress(0, len(filenames))
    for i, img_path in enumerate(filenames):
        img = io.imread(img_path)
        img_features = []
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)

        channels = helpers.get_channels(img_resize)

        # mean
        for channel in channels:
            img_features.append(np.mean(channel))

        # var
        for channel in channels:
            img_features.append(np.var(channel))

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

def plot_features(features, title, name_of_feature, save=True, show=True):
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
