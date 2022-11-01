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
#dataset.dataset_examples_each_class(data_train_dir, img_height, img_width, True)
#dataset.dataset_distribution(data_train_dir)

train_features = features.get_features(dataset_train, img_height, img_width)
print(np.shape(train_features))
print(train_features)

