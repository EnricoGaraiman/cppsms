import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans
from numpy.linalg import norm
import src.helpers as helpers

def extract_descriptors(dataset_images, type='SIFT'):
    if type == 'SIFT':
        extractor = cv2.SIFT_create()
    elif type == 'ORB':
        extractor = cv2.ORB_create()

    keypoints = []
    descriptors = []

    helpers.progress(0, len(dataset_images))
    for i, img in enumerate(dataset_images):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # extract keypoints and descriptors for each image
        if type == 'SIFT':
            img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        elif type == 'ORB':
            img_keypoints = extractor.detect(img,None)
            # compute the descriptors with ORB
            img_keypoints, img_descriptors = extractor.compute(img, img_keypoints)
        keypoints.append(img_keypoints)
        descriptors.append(img_descriptors.astype('float'))
        helpers.progress(i, len(dataset_images), type + ' descriptors')

    # output_image = []
    # for x in range(3):
    #     output_image.append(
    #         cv2.drawKeypoints(cv2.normalize(dataset_images[x], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),
    #                           keypoints[x], None, color=(0, 255, 0), flags=0))
    #     plt.imshow(output_image[x], cmap='gray')
    #     plt.show()

    return keypoints, descriptors

def extract_bovw(keypoints, descriptors, number_of_images):

    # compute Kmeans
    all_descriptors = []
    for img_descriptors in descriptors:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    all_descriptors = np.stack(all_descriptors)

    k = 120
    codebook, variance = kmeans(all_descriptors, k)

    visual_words = []
    helpers.progress(0, len(descriptors))
    for i, img_descriptors in enumerate(descriptors):
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
        helpers.progress(i, len(descriptors), 'BOW')

    tfidf, frequency_vectors = get_frequency_vector(visual_words, k, number_of_images)

    return tfidf, frequency_vectors

def get_frequency_vector(visual_words, k, number_of_images):
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    frequency_vectors = np.stack(frequency_vectors)

    # plt.bar(list(range(k)), frequency_vectors[0])
    # plt.show()

    # tf-idf
    # df is the number of images that a visual word appears in
    # we calculate it by counting non-zero values as 1 and summing
    df = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(number_of_images / df)
    tfidf = frequency_vectors * idf

    return tfidf, frequency_vectors

def get_similar_images(tfidf, images, search_imgs, show=False, save=False):
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

