import numpy as np
import src.helpers as helpers
from skimage import io
from skimage.transform import resize



def get_features(dataset,img_height, img_width):
    features = []
    for img_path in dataset.filenames:
        img = io.imread(img_path)
        img_features = []
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)
        for channel in helpers.get_channels(img_resize):
            img_features.append(np.mean(channel))

        features.append(img_features)

    return features



