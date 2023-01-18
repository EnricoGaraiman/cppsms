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

    if count == total:
        print()


rgb2gray = lambda img: np.uint8(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.144 * img[:, :, 2])

mse = lambda x, y: np.mean((x - y) ** 2)





# import splitfolders
#
# input_folder = 'stanfordDogsDataset/images'
# output_folder = 'stanfordDogsDataset/split_images'
# splitfolders.ratio(input_folder, output = output_folder, seed = 1337, ratio = (0.8, 0, 0.2))





# import os
# import glob
#
# # files = glob.glob('stanfordDogsDataset/split_images_copy/train/*/*')
# # files = glob.glob('stanfordDogsDataset/split_images_copy/test/*/*')
# # print(files)
# for i in dataset_train.target_names:
#     for f in glob.glob('stanfordDogsDataset/split_images_copy/test/' + i + '/*')[2:]:
#         os.remove(f)


# # read and save images cropped
# import matplotlib.image as mpimg
# import os
# import glob
#
# for i in dataset_train.target_names:
#     for f in glob.glob('stanfordDogsDataset/split_images/train/' + i + '/*'):
#         if not os.path.exists('stanfordDogsDataset/split_images_crop/train/' + i):
#             os.makedirs('stanfordDogsDataset/split_images_crop/train/' + i)
#         img = mpimg.imread(f)
#         text = open(('stanfordDogsDataset/Annotation/' + f.split('/')[-1]).replace('\\', '/').replace('.jpg', '')).read()
#         x_min = int(text.split('<xmin>')[1].split('</xmin>')[0])
#         x_max = int(text.split('<xmax>')[1].split('</xmax>')[0])
#         y_min = int(text.split('<ymin>')[1].split('</ymin>')[0])
#         y_max = int(text.split('<ymax>')[1].split('</ymax>')[0])
#         img = img[y_min:y_max, x_min:x_max]
#         mpimg.imsave(('stanfordDogsDataset/split_images_crop/train/' + f.split('/')[-1]).replace('\\', '/'), img)
#
# for i in dataset_test.target_names:
#     for f in glob.glob('stanfordDogsDataset/split_images/test/' + i + '/*'):
#         if not os.path.exists('stanfordDogsDataset/split_images_crop/test/' + i):
#             os.makedirs('stanfordDogsDataset/split_images_crop/test/' + i)
#         img = mpimg.imread(f)
#         text = open(('stanfordDogsDataset/Annotation/' + f.split('/')[-1]).replace('\\', '/').replace('.jpg', '')).read()
#         x_min = int(text.split('<xmin>')[1].split('</xmin>')[0])
#         x_max = int(text.split('<xmax>')[1].split('</xmax>')[0])
#         y_min = int(text.split('<ymin>')[1].split('</ymin>')[0])
#         y_max = int(text.split('<ymax>')[1].split('</ymax>')[0])
#         img = img[y_min:y_max, x_min:x_max]
#         mpimg.imsave(('stanfordDogsDataset/split_images_crop/test/' + f.split('/')[-1]).replace('\\', '/'), img)
