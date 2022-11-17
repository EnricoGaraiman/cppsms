import sys
import numpy as np

def get_channels(img):
    return [img[:, :, 0], img[:, :, 1], img[:, :, 2]]

# Print iterations progress https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s | %s\r' % (bar, percents, '%', suffix))


rgb2gray = lambda img: np.uint8(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.144 * img[:, :, 2])

mse = lambda x, y: np.mean((x - y) ** 2)





# import splitfolders
#
# input_folder = 'stanfordDogsDataset/images'
# output_folder = 'stanfordDogsDataset/split_images'
# splitfolders.ratio(input_folder, output = output_folder, seed = 1337, ratio = (0.8, 0, 0.2))