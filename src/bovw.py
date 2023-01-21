import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import src.helpers as helpers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_bovw(dataset_images, number_of_clusters, redo_features, redo_suffix = '', extractor = 'SIFT'):
    """
    The function receives the images and the number of clusters. compute all descriptors using SIFT/ORB,
    group them, extract features and standardize them. The kmeans object,
    the scale object and the standardized features are returned.

    @param dataset_images:
    @param number_of_clusters:
    @param redo_features:
    @param redo_suffix:
    @param extractor:
    @return:
        kmeans
        scale
        features
        frequency
    """
    if not redo_features:
        return None, None, np.load('results/features/features_train' + redo_suffix + '.npy')

    # descriptor_list = []
    # for img in dataset_images:
    #     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #     des = sift_descriptor(img)
    #     descriptor_list.append(des)

    count = 0
    descriptor_list = []
    helpers.progress(0, len(dataset_images))
    for i, img in enumerate(dataset_images):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        if extractor == 'SIFT':
            des = sift_descriptor(img)
        elif extractor == 'ORB':
            des = orb_descriptor(img)
        if des is not None:
            count += 1
            descriptor_list.append(des)
        else:
            print('train', i)
        helpers.progress(i + 1, len(dataset_images), extractor + ' train')

    descriptors = descriptor_vstack(descriptor_list)
    print("\nDescriptors vstacked.")

    kmeans = descriptors_clustering(descriptors, number_of_clusters)
    print("Descriptors clustered using K-means.")

    im_features = features_extract(kmeans, descriptor_list, count, number_of_clusters, extractor)
    print("Images features extracted using " + extractor)

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Features Histogram for the given clusters.")
    plot_histogram(im_features, number_of_clusters)

    return kmeans, scale, im_features


def sift_descriptor(img):
    """
       The function receives the image and return SIFT descriptors.

       @param img:
       @return:
           descriptors
    """
    sift = cv2.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)

    # # show
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img)
    # plt.show()
    # plt.clf()

    return descriptors

def orb_descriptor(img):
    """
       The function receives the image and return ORB descriptors.

       @param img:
       @return:
           descriptors
    """
    sift = cv2.ORB_create()
    kp, descriptors = sift.detectAndCompute(img, None)

    # # show
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img)
    # plt.show()
    # plt.clf()

    return descriptors


def descriptor_vstack(descriptor_list):
    """
       The function receives the list of descriptors and return a stacked list.

       @param descriptor_list:
       @return:
           descriptors
    """
    helpers.progress(0, len(descriptor_list))
    descriptors = np.array(descriptor_list[0])
    for i, descriptor in enumerate(descriptor_list[1:]):
        descriptors = np.vstack((descriptors, descriptor))
        helpers.progress(i + 1, len(descriptor_list), 'Descriptor vstack')

    return descriptors


def descriptors_clustering(descriptors, no_clusters):
    """
       The function receives descriptors and number of clusters.
       KMeans training is performed and the object is returned.

       @param descriptors:
       @param no_clusters:
       @return:
           kmeans
    """
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans


def features_extract(kmeans, descriptor_list, image_count, no_clusters, extractor):
    """
       The function receives the kmeans, list of descriptors, number of images and number of clusters.
       Features are returned after kmeans predict.

       @param kmeans:
       @param descriptor_list:
       @param image_count:
       @param no_clusters:
       @param extractor:
       @return:
           im_features
    """
    size = 128 if extractor == 'SIFT' else 32
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    helpers.progress(0, image_count)
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, size)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
        helpers.progress(i + 1, image_count, 'Feature extraction')

    return im_features


def plot_histogram(im_features, no_clusters):
    """
       The function receives the features and the number of clusters.
       The histogram of the generated visual words is made, where the frequency is on y,
       and the index of the word on x.

       @param im_features:
       @param no_clusters:
    """
    plt.figure(figsize=(18, 9))
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])
    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Complete Visual Words Generated", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/histogram-bow' + str(no_clusters) + '.png')
    plt.clf()


def extract_test_bovw(dataset_test_images, number_of_clusters, kmeans, scale, redo_features, redo_suffix = '', extractor = 'SIFT'):
    """
       The function receives the set of test images, the number of clusters, the kmeans object and the scales object.
       The descriptors are extracted and then the features of the test set are determined which are also standardized.

       @param dataset_test_images:
       @param number_of_clusters:
       @param kmeans:
       @param scale:
       @param redo_features:
       @param extractor:
       @return:
           test_features
           frequency
    """
    if not redo_features:
        return np.load('results/features/features_test' + redo_suffix + '.npy')

    count = 0
    descriptor_list = []
    helpers.progress(0, len(dataset_test_images))
    for i, img in enumerate(dataset_test_images):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        if extractor == 'SIFT':
            des = sift_descriptor(img)
        elif extractor == 'ORB':
            des = orb_descriptor(img)
        if des is not None:
            count += 1
            descriptor_list.append(des)
        else:
            print('test', i)
        helpers.progress(i + 1, len(dataset_test_images), extractor + ' test')

    test_features = features_extract(kmeans, descriptor_list, count, number_of_clusters, extractor)
    test_features = scale.transform(test_features)

    return test_features


def get_tfidf(visual_words):
    """
       The function receives the frequency train and number of images and return tfidf

       @param visual_words:
       @return:
           tfidf
    """
    # plt.figure()
    # plt.bar(list(range(number_of_clusters)), visual_words[0])
    # plt.show()
    # plt.clf()

    df = np.sum(visual_words > 0, axis=0)
    idf = np.log(len(visual_words) / df)
    tfidf = visual_words * idf

    # plt.figure()
    # plt.bar(list(range(number_of_clusters)), tfidf[0])
    # plt.show()
    # plt.clf()

    return tfidf


def get_similar_images(features_train, images_train, features_test, images_test, dataset_train_labels, dataset_test_labels, show=False, save=False, no_clusters = None):
    """
       The function receives tfidf and the set of images. For given search images, find similar top_k images

       @param features_train:
       @param images_train:
       @param features_test:
       @param images_test:
       @param dataset_train_labels:
       @param dataset_test_labels:
       @param show:
       @param save:
       @param no_clusters:
    """
    top_k = 6

    correct = 0
    for j, img in enumerate(images_test):
        # get search image vector
        a = features_test[j]
        b = features_train  # set search space to the full sample

        # get the cosine distance for the search image `a`
        cosine_similarity = np.dot(a, b.T) / (norm(a) * norm(b, axis=1))

        # get the top k indices for most similar vecs
        idx = np.argsort(-cosine_similarity)[:top_k]

        # display the results
        good = False
        if show or save:
            plt.figure(figsize=(10, 5))
        for i, index in enumerate(idx):
            if show or save:
                if i == 0:
                    plt.subplot(2, 2, 1)
                    plt.gca().set_title(f"{j}: Original image")
                    plt.imshow(img, cmap='gray')
                else:
                    if i == 1:
                        plt.subplot(2, 2, 2)
                        plt.gca().set_title(f"{index}: {round(cosine_similarity[index], 4)}")
                        plt.imshow(images_train[index], cmap='gray')
            if i == 1 and dataset_test_labels[j] == dataset_train_labels[index]:
                correct = correct + 1
                good = True

        if show:
            plt.show()
        if save:
            plt.savefig('results/search_bow/' + str(no_clusters) + 'search_engine_' + str(j) + ('_correct' if good else '') + '.png')
        plt.clf()

    print('Accuracy image search: ', correct / len(images_test) * 100, '%')
