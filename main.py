import src.dataset as dataset
import src.features as features
import numpy as np
import src.knn as knn
import src.bovw as bovw
import random

# ___________________________________________
# PARAMETERS
# ___________________________________________

data_train_dir = 'stanfordDogsDataset/split_images_small2/train'
data_test_dir = 'stanfordDogsDataset/split_images_small2/test'
img_height = 128
img_width = 128
redo_features = True

# ___________________________________________
# DATASET
# ___________________________________________

dataset_train, dataset_test, class_names = dataset.load_dataset_paths(data_train_dir, data_test_dir)
# dataset.dataset_examples_each_class(data_train_dir, img_height, img_width, True)
# dataset.dataset_distribution(data_train_dir)

if redo_features:
   # ___________________________________________
    # FEATURES ALL DATASET
    # ___________________________________________

    # train_features = features.get_features(dataset_train.filenames, img_height, img_width)
    # test_features = features.get_features(dataset_test.filenames, img_height, img_width)

    # features.plot_features(train_features[:, 0:3], 'Dataset mean', 'Mean', True, False)
    # features.plot_features(train_features[:, 3:6], 'Dataset variance', 'Variance', True, False)
    # features.plot_features(train_features[:, 6:9], 'Dataset train skewness', 'Skewness', False, True)
    # features.plot_features(train_features[:, 9:12], 'Dataset train kurtosis', 'Kurtosis', False, True)
    # features.plot_moments_hu(train_features[:, 6:13], 'Dataset moments hu', 'Moments hu', True, False)

    # ___________________________________________
    # FEATURES BY CLASSES (USE FOR PLOT REPRESENTATION)
    # ___________________________________________

    # train_classes_features = features.get_features_classes(data_train_dir, img_height, img_width)

    # features.plot_features_by_classes(train_classes_features[:, 0:3], 'Dataset mean by classes', 'Class Mean', True, False)
    # features.plot_features_by_classes(train_classes_features[:, 3:6], 'Dataset variance by classes', 'Class Variance', True, False)
    # features.plot_features_by_classes(train_classes_features[:, 6:9], 'Dataset train skewness by classes', 'Class Skewness', False, True)
    # features.plot_features_by_classes(train_classes_features[:, 9:12], 'Dataset train kurtosis by classes', 'Class Kurtosis', False, True)

    dataset_train_images = dataset.load_dataset(dataset_train.filenames,  False, False, img_height, img_width)
    dataset_test_images = dataset.load_dataset(dataset_test.filenames, False, False, img_height, img_width)

    dataset_train_labels = dataset_train.target
    dataset_test_labels = dataset_test.target

    # dataset.display_img_by_index(dataset_train_images, np.argmax(train_features[:, 0:3].sum(axis=1)), train_features[:, 0:3], 'Image with biggest mean', True, False)
    # dataset.display_img_by_index(dataset_train_images, np.argmin(train_features[:, 0:3].sum(axis=1)), train_features[:, 0:3], 'Image with smallest mean', True, False)
    # dataset.display_img_by_index(dataset_train_images, np.argmax(train_features[:, 3:6].sum(axis=1)), train_features[:, 3:6], 'Image with biggest var', True, False)
    # dataset.display_img_by_index(dataset_train_images, np.argmin(train_features[:, 3:6].sum(axis=1)), train_features[:, 3:6], 'Image with smallest var', True, False)

    # ___________________________________________
    # COVARIATION ANALYSIS
    # ___________________________________________

    # covariation_matrix = features.get_covariation_matrix(dataset_train.filenames, img_height, img_width)
    # features.view_images_with_max_covariation(covariation_matrix, dataset_train.filenames, img_height, img_width)

    # dubios
    # covariances_train = features.get_covariances_matrix_for_each_images(dataset_train.filenames, img_height, img_width)
    # covariances_test = features.get_covariances_matrix_for_each_images(dataset_test.filenames, img_height, img_width)

    # dubios 2
    # covariances_train = features.get_covariances_pixel_with_pixel(dataset_train.filenames, img_height, img_width)

    # ___________________________________________
    # BAG OF VISUAL WORDS
    # ___________________________________________

    # for train
    keywords_train, descriptors_train, index_img_with_no_descriptors_train = bovw.extract_descriptors(dataset_train_images, 'SIFT')
    tfidf_train, frequency_vectors_train, features_train = bovw.extract_bovw(descriptors_train, len(dataset_train_labels), dataset_train_images)

    # for test
    keywords_test, descriptors_test, index_img_with_no_descriptors_test = bovw.extract_descriptors(dataset_test_images, 'SIFT')
    tfidf_test, frequency_vectors_test, features_test = bovw.extract_bovw(descriptors_test, len(dataset_test_labels))

    # remove images with no descriptors
    print(index_img_with_no_descriptors_train, index_img_with_no_descriptors_test)
    dataset_train_images = [ele for idx, ele in enumerate(dataset_train_images) if idx not in index_img_with_no_descriptors_train]
    dataset_test_images = [ele for idx, ele in enumerate(dataset_test_images) if idx not in index_img_with_no_descriptors_test]
    dataset_train_labels = [ele for idx, ele in enumerate(dataset_train_labels) if idx not in index_img_with_no_descriptors_train]
    dataset_test_labels = [ele for idx, ele in enumerate(dataset_test_labels) if idx not in index_img_with_no_descriptors_test]

    # save features on disk
    dataset.save_features([tfidf_train, descriptors_train, frequency_vectors_train, dataset_train_labels, features_train], ['tfidf_train', 'descriptors_train', 'frequency_vectors_train', 'dataset_train_labels', 'features_train'])
    dataset.save_features([tfidf_test, descriptors_test, frequency_vectors_test, dataset_test_labels, features_test], ['tfidf_test', 'descriptors_test', 'frequency_vectors_test', 'dataset_test_labels', 'features_test'])

    # get similar images
    # search_imgs = random.sample(range(0, len(dataset_train_images)), 50)
    # bovw.get_similar_images(tfidf_train, dataset_train_images, search_imgs, True, True)

else:
    # load features from disk
    print('Start load features from disk')
    tfidf_train, descriptors_train, frequency_vectors_train, dataset_train_labels, features_train = dataset.load_features_train()
    tfidf_test, descriptors_test, frequency_vectors_test, dataset_test_labels, features_test = dataset.load_features_test()
    print('Loaded features from disk')

# ___________________________________________
# CLASSIFICATION
# ___________________________________________

# # standardize data
# dataset_train_reduced_standardize = dataset_train_reduced.copy()
# dataset_test_reduced_standardize = dataset_test_reduced.copy()
# dataset_train_reduced_standardize = (dataset_train_reduced_standardize - np.mean(dataset_train_reduced_standardize)) / np.std(dataset_train_reduced_standardize)
# dataset_test_reduced_standardize = (dataset_test_reduced_standardize - np.mean(dataset_test_reduced_standardize)) / np.std(dataset_test_reduced_standardize)
#
# # normalize data
# dataset_train_reduced_norm = dataset_train_reduced_standardize.copy()
# dataset_test_reduced_norm = dataset_test_reduced_standardize.copy()
# dataset_train_reduced_norm /= np.max(np.abs(dataset_train_reduced_norm))
# dataset_test_reduced_norm /= np.max(np.abs(dataset_test_reduced_norm))

knn.knn_classifier(features_train, dataset_train_labels, features_test, dataset_test_labels, class_names)


# ___________________________________________
# SEARCH ENGINE
# ___________________________________________

# features.search_similar_image(train_features, dataset_train.filenames, dataset_test.filenames)
