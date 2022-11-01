import numpy as np
import src.helpers as helpers
from skimage import io
from skimage.transform import resize
from scipy.stats import skew,kurtosis
import matplotlib.pyplot as plt



def get_features(dataset,img_height, img_width):
    features = []
    for img_path in dataset.filenames[0:100]:
        img = io.imread(img_path)
        img_features = []
        img_resize = resize(img, (img_height, img_width), anti_aliasing=True)

        channels = helpers.get_channels(img_resize)

        #mean
        for channel in channels:
            img_features.append(np.mean(channel))

        #var
        for channel in channels:
            img_features.append(np.var(channel))

        #skew = asimetrie
        for channel in channels:
            img_features.append(skew(channel, axis=1, bias=False))

        #kurtosis = curtoza
        for channel in channels:
            img_features.append(kurtosis(channel, fisher=False))

        features.append(img_features)

    return np.array(features, dtype=object)

def plot_features(features, title, name_of_feature, save=True, show=True):
    fig = plt.figure()
    plt.title(title)

    r = features[:, 0]
    g = features[:, 1]
    b = features[:, 2]

    print(r)

    r.sort()
    g.sort()
    b.sort()

    plt.plot(r)
    plt.plot(g)
    plt.plot(b)
    plt.ylabel(name_of_feature)
    plt.xlabel('Number of image')
    plt.legend(['R channel', 'G channel', 'B channel'])

    if show:
        plt.show()
    if save:
        fig.savefig('results/' + title + '.jpg')







