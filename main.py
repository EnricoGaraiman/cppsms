import src.dataset as dataset
import src.features as features
import numpy as np
import src.knn as knn
import src.bovw as bovw

# ___________________________________________
# PARAMETERS
# ___________________________________________

data_train_dir = 'stanfordDogsDataset/database/train'
data_test_dir = 'stanfordDogsDataset/database/test'
img_height = 256
img_width = 256
redo_features = False
redo_suffix = '_10'
bbox = False
no_clusters = 40
extractor = 'SIFT'

# ___________________________________________
# DATASET
# ___________________________________________

dataset_train, dataset_test, class_names = dataset.load_dataset_paths(data_train_dir, data_test_dir)
dataset.dataset_examples_each_class(data_train_dir, img_height, img_width, True)
dataset.dataset_distribution(data_train_dir)

# ___________________________________________
# FEATURES ALL DATASET
# ___________________________________________

features_train = features.get_features(dataset_train.filenames, img_height, img_width)
features_test = features.get_features(dataset_test.filenames, img_height, img_width)
features_train, features_test = features.standardize(features_train, features_test)

features.plot_features(features_train[:, 0:3], 'Dataset mean', 'Mean', True, False)
features.plot_features(features_train[:, 3:6], 'Dataset variance', 'Variance', True, False)
# features.plot_features(features_train[:, 6:9], 'Dataset train skewness', 'Skewness', False, True)
# features.plot_features(features_train[:, 9:12], 'Dataset train kurtosis', 'Kurtosis', False, True)
features.plot_moments_hu(features_train[:, 6:13], 'Dataset moments hu', 'Moments hu', True, False)

# ___________________________________________
# FEATURES BY CLASSES (USE FOR PLOT REPRESENTATION)
# ___________________________________________

train_classes_features = features.get_features_classes(data_train_dir, img_height, img_width)

features.plot_features_by_classes(train_classes_features[:, 0:3], 'Dataset mean by classes', 'Class Mean', True, False)
features.plot_features_by_classes(train_classes_features[:, 3:6], 'Dataset variance by classes', 'Class Variance', True, False)
# features.plot_features_by_classes(train_classes_features[:, 6:9], 'Dataset train skewness by classes', 'Class Skewness', False, True)
# features.plot_features_by_classes(train_classes_features[:, 9:12], 'Dataset train kurtosis by classes', 'Class Kurtosis', False, True)

dataset_train_images = dataset.load_dataset(dataset_train.filenames, bbox, False, img_height, img_width)
dataset_test_images = dataset.load_dataset(dataset_test.filenames, bbox, False, img_height, img_width)

dataset_train_labels = dataset_train.target
dataset_test_labels = dataset_test.target

dataset.display_img_by_index(dataset_train_images, np.argmax(features_train[:, 0:3].sum(axis=1)),
                             features_train[:, 0:3], 'Image with biggest mean', True, False)
dataset.display_img_by_index(dataset_train_images, np.argmin(features_train[:, 0:3].sum(axis=1)),
                             features_train[:, 0:3], 'Image with smallest mean', True, False)
dataset.display_img_by_index(dataset_train_images, np.argmax(features_train[:, 3:6].sum(axis=1)),
                             features_train[:, 3:6], 'Image with biggest var', True, False)
dataset.display_img_by_index(dataset_train_images, np.argmin(features_train[:, 3:6].sum(axis=1)),
                             features_train[:, 3:6], 'Image with smallest var', True, False)

# ___________________________________________
# COVARIANCE ANALYSIS
# ___________________________________________

covariance_matrix = features.get_covariance_matrix(dataset_train.filenames, img_height, img_width)
features.view_images_with_max_covariance(covariance_matrix, dataset_train.filenames, img_height, img_width)

# ___________________________________________
# BAG OF VISUAL WORDS
# ___________________________________________

# extract BOW for training
kmeans, scale, features_train = bovw.extract_bovw(dataset_train_images, no_clusters, redo_features, redo_suffix,
                                                  extractor)

# optimal k determination
optimal_k = knn.get_optimal_k_for_knn(features_train, dataset_train_labels, no_clusters)

# extract BOW for testing
features_test = bovw.extract_test_bovw(dataset_test_images, no_clusters, kmeans, scale, redo_features, redo_suffix,
                                       extractor)

# compute tfidf
tfidf_train = bovw.get_tfidf(features_train)
tfidf_test = bovw.get_tfidf(features_test)

# save features
if redo_features:
    dataset.save_features([features_train, features_test], ['features_train' + redo_suffix + '_' + str(no_clusters),
                                                            'features_test' + redo_suffix + '_' + str(no_clusters)])

# ___________________________________________
# CLASSIFICATION
# ___________________________________________

knn.knn_classifier(features_train, dataset_train_labels, features_test, dataset_test_labels, class_names, optimal_k,
                   no_clusters)

# ___________________________________________
# SEARCH ENGINE
# ___________________________________________

# get similar images based on statistical moments + glcm
features.search_similar_image(features_train, dataset_train.filenames, dataset_test.filenames, dataset_train_labels,
                              dataset_test_labels)

# get similar images based on BOVW
bovw.get_similar_images(features_train, dataset_train_images, features_test, dataset_test_images, dataset_train_labels,
                        dataset_test_labels, False, True, no_clusters)
