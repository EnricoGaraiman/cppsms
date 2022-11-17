import src.dataset as dataset
import src.features as features
import numpy as np

# ___________________________________________
# PARAMETERS
# ___________________________________________

data_train_dir = 'stanfordDogsDataset/split_images/train'
data_test_dir = 'stanfordDogsDataset/split_images/test'
img_height = 128
img_width = 128

# ___________________________________________
# DATASET
# ___________________________________________

dataset_train, dataset_test = dataset.load_dataset_paths(data_train_dir, data_test_dir)
# dataset.dataset_examples_each_class(data_train_dir, img_height, img_width, True)
# dataset.dataset_distribution(data_train_dir)

# ___________________________________________
# FEATURES ALL DATASET
# ___________________________________________

# train_features = features.get_features(dataset_train.filenames, img_height, img_width)

#
# features.plot_features(train_features[:, 0:3], 'Dataset mean', 'Mean', True, False)
# features.plot_features(train_features[:, 3:6], 'Dataset variance', 'Variance', True, False)
# features.plot_features(train_features[:, 6:9], 'Dataset train skewness', 'Skewness', False, True)
# features.plot_features(train_features[:, 9:12], 'Dataset train kurtosis', 'Kurtosis', False, True)
# features.plot_moments_hu(train_features[:, 6:13], 'Dataset moments hu', 'Moments hu', True, False)



# ___________________________________________
# FEATURES BY CLASSES (USE FOR PLOT REPRESENTATION)
# ___________________________________________

# train_classes_features = features.get_features_classes(data_train_dir, img_height, img_width)
#
# features.plot_features_by_classes(train_classes_features[:, 0:3], 'Dataset mean by classes', 'Class Mean', True, False)
# features.plot_features_by_classes(train_classes_features[:, 3:6], 'Dataset variance by classes', 'Class Variance', True, False)
# features.plot_features_by_classes(train_classes_features[:, 6:9], 'Dataset train skewness by classes', 'Class Skewness', False, True)
# features.plot_features_by_classes(train_classes_features[:, 9:12], 'Dataset train kurtosis by classes', 'Class Kurtosis', False, True)



# dataset_train_images = dataset.load_dataset(dataset_train.filenames, True, False)
# dataset.display_img_by_index(dataset_train_images, np.argmax(train_features[:, 0:3].sum(axis=1)), train_features[:, 0:3], 'Image with biggest mean', True, False)
# dataset.display_img_by_index(dataset_train_images, np.argmin(train_features[:, 0:3].sum(axis=1)), train_features[:, 0:3], 'Image with smallest mean', True, False)
# dataset.display_img_by_index(dataset_train_images, np.argmax(train_features[:, 3:6].sum(axis=1)), train_features[:, 3:6], 'Image with biggest var', True, False)
# dataset.display_img_by_index(dataset_train_images, np.argmin(train_features[:, 3:6].sum(axis=1)), train_features[:, 3:6], 'Image with smallest var', True, False)



# ___________________________________________
# CORRELATION ANALYSIS
# ___________________________________________
# correlation_matrix = features.get_correlation_matrix(dataset_train.filenames, img_height, img_width)
# features.view_images_with_max_correlation(correlation_matrix, dataset_train.filenames, img_height, img_width)

# dubios
# correlations = features.get_correlations_matrix_for_each_images(dataset_train.filenames, img_height, img_width)


# ___________________________________________
# PCA ANALYSIS
# ___________________________________________



# ___________________________________________
# SEARCH ENGINE
# ___________________________________________
# features.search_similar_image(train_features, dataset_train.filenames, dataset_test.filenames)
