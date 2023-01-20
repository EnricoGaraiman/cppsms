import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import src.helpers as helpers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_bovw(dataset_images, number_of_clusters, redo_features, redu_suffix = ''):
    """
    The function receives the images and the number of clusters. compute all descriptors using SIFT,
    group them, extract features and standardize them. The kmeans object,
    the scale object and the standardized features are returned.

    @param dataset_images:
    @param number_of_clusters:
    @param redo_features:
    @param redu_suffix:
    @return:
        kmeans
        scale
        features
        frequency
    """
    if not redo_features:
        return None, None, np.load('results/features/features_train' + redu_suffix + '.npy')

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
        des = sift_descriptor(img)
        if des is not None:
            count += 1
            descriptor_list.append(des)
        else:
            print('train', i)
        helpers.progress(i + 1, len(dataset_images), 'SIFT train')

    descriptors = descriptor_vstack(descriptor_list)
    print("\nDescriptors vstacked.")

    kmeans = descriptors_clustering(descriptors, number_of_clusters)
    print("Descriptors clustered using K-means.")

    im_features = features_extract(kmeans, descriptor_list, count, number_of_clusters)
    print("Images features extracted using SIFT.")

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

    # show
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img)
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


def features_extract(kmeans, descriptor_list, image_count, no_clusters):
    """
       The function receives the kmeans, list of descriptors, number of images and number of clusters.
       Features are returned after kmeans predict.

       @param kmeans:
       @param descriptor_list:
       @param image_count:
       @param no_clusters:
       @return:
           im_features
    """
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    helpers.progress(0, image_count)
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
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
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Visual Words Generated")
    plt.xticks(x_scalar + 0.4, x_scalar, fontsize=10)
    # plt.show()
    plt.savefig('results/histogram-bow.png')
    plt.clf()


def extract_test_bovw(dataset_test_images, number_of_clusters, kmeans, scale, redo_features, redu_suffix = ''):
    """
       The function receives the set of test images, the number of clusters, the kmeans object and the scales object.
       The descriptors are extracted and then the features of the test set are determined which are also standardized.

       @param dataset_test_images:
       @param number_of_clusters:
       @param kmeans:
       @param scale:
       @param redo_features:
       @return:
           test_features
           frequency
    """
    if not redo_features:
        return np.load('results/features/features_test' + redu_suffix + '.npy')

    count = 0
    descriptor_list = []
    helpers.progress(0, len(dataset_test_images))
    for i, img in enumerate(dataset_test_images):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        des = sift_descriptor(img)
        if des is not None:
            count += 1
            descriptor_list.append(des)
        else:
            print('test', i)
        helpers.progress(i + 1, len(dataset_test_images), 'SIFT test')

    test_features = features_extract(kmeans, descriptor_list, count, number_of_clusters)
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


def get_similar_images(tfidf, images, search_imgs, show=False, save=False):
    """
       The function receives tfidf and the set of images. For given search images, find similar top_k images

       @param tfidf:
       @param images:
       @param search_imgs:
       @param show:
       @param save:
    """
    top_k = 6

    for ind in search_imgs:
        # get search image vector
        a = tfidf[ind]
        b = tfidf  # set search space to the full sample

        # get the cosine distance for the search image `a`
        cosine_similarity = np.dot(a, b.T) / (norm(a) * norm(b, axis=1))

        # get the top k indices for most similar vecs
        idx = np.argsort(-cosine_similarity)[:top_k]

        # display the results
        plt.figure(figsize=(10, 5))
        for i, index in enumerate(idx):
            if i == 0:
                plt.subplot(2, top_k, top_k // 2)
                plt.gca().set_title(f"{ind}: Original image: {round(cosine_similarity[index], 4)}")
                plt.imshow(images[index], cmap='gray')
            else:
                plt.subplot(2, top_k, top_k // 2 + 3 + i)
                plt.gca().set_title(f"{index}: {round(cosine_similarity[index], 4)}")
                plt.imshow(images[index], cmap='gray')

        if show:
            plt.show()
        if save:
            plt.savefig('results/search_bow/search_engine_' + str(ind) + '.png')
        plt.clf()

#
# def extract_bovw(descriptors, number_of_images, dataset_images = None):
#
#     # compute Kmeans
#     all_descriptors = []
#     for img_descriptors in descriptors:
#         for descriptor in img_descriptors:
#             all_descriptors.append(descriptor)
#     all_descriptors = np.stack(all_descriptors)
#
#     k = 200
#     codebook, variance = kmeans(all_descriptors, k, 5)
#
#     visual_words = []
#     im_features = np.zeros((len(descriptors), k), "float32")
#     helpers.progress(0, len(descriptors))
#     for i, img_descriptors in enumerate(descriptors):
#         print(np.shape(img_descriptors))
#         print(np.shape(img_descriptors[1]))
#         img_visual_words, distance = vq(img_descriptors, codebook)
#         print(np.shape(img_visual_words))
#         visual_words.append(img_visual_words)
#         for w in img_visual_words:
#             print(np.shape(w), 'da')
#             im_features[i][w] += 1
#         if dataset_images:
#             plt.imshow(dataset_images[1])
#             plt.show()
#         helpers.progress(i, len(descriptors), 'BOW')
#
#     tfidf, frequency_vectors = get_frequency_vector(visual_words, k, number_of_images)
#
#     return tfidf, frequency_vectors, im_features

# def get_frequency_vector(visual_words, k, number_of_images):
#     frequency_vectors = []
#     for img_visual_words in visual_words:
#         img_frequency_vector = np.zeros(k)
#         for word in img_visual_words:
#             img_frequency_vector[word] += 1
#         frequency_vectors.append(img_frequency_vector)
#     frequency_vectors = np.stack(frequency_vectors)
#
#     # plt.bar(list(range(k)), frequency_vectors[0])
#     # plt.show()
#
#     # tf-idf
#     # df is the number of images that a visual word appears in
#     # we calculate it by counting non-zero values as 1 and summing
#     df = np.sum(frequency_vectors > 0, axis=0)
#     idf = np.log(number_of_images / df)
#     tfidf = frequency_vectors * idf
#
#     return tfidf, frequency_vectors
#
#
