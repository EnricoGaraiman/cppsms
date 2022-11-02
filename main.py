import src.dataset as dataset
import src.features as features
import numpy as np

# ___________________________________________
# PARAMETERS
# ___________________________________________

data_train_dir = 'stanfordDogsDataset/images'
data_test_dir = 'stanfordDogsDataset/images'
img_height = 200
img_width = 200

# ___________________________________________
# DATASET
# ___________________________________________

dataset_train, dataset_test = dataset.load_dataset_paths(data_train_dir, data_test_dir)
# dataset.dataset_examples_each_class(data_train_dir, img_height, img_width, True)
# dataset.dataset_distribution(data_train_dir)

# ___________________________________________
# FEATURES ALL DATASET
# ___________________________________________

train_features = features.get_features(dataset_train.filenames, img_height, img_width)

features.plot_features(train_features[:, 0:3], 'Dataset train mean', 'Mean', True, False)
features.plot_features(train_features[:, 3:6], 'Dataset train variance', 'Variance', True, False)
# features.plot_features(train_features[:, 6:9], 'Dataset train skewness', 'Skewness', False, True)
# features.plot_features(train_features[:, 9:12], 'Dataset train kurtosis', 'Kurtosis', False, True)

# ___________________________________________
# FEATURES BY CLASSES (USE FOR PLOT REPRESENTATION)
# ___________________________________________

train_classes_features = features.get_features_classes(data_train_dir, img_height, img_width)

features.plot_features_by_classes(train_classes_features[:, 0:3], 'Dataset train mean by classes', 'Class Mean', True, False)
features.plot_features_by_classes(train_classes_features[:, 3:6], 'Dataset train variance by classes', 'Class Variance', True, False)
# features.plot_features_by_classes(train_classes_features[:, 6:9], 'Dataset train skewness by classes', 'Class Skewness', False, True)
# features.plot_features_by_classes(train_classes_features[:, 9:12], 'Dataset train kurtosis by classes', 'Class Kurtosis', False, True)

# ___________________________________________
# COVARIANCE ANALYSIS
# ___________________________________________
