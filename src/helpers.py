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