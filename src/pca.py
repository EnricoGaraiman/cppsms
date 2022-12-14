from sklearn.decomposition import PCA
import src.helpers as helpers
import matplotlib.pyplot as plt
import numpy as np


def choose_pca_components(n_components, dataset_train_images):
    pca_train = PCA(n_components=n_components)  # limitare sklearn n_comp < nr_img !
    pca_train.fit(dataset_train_images)

    plt.grid()
    plt.plot(np.cumsum(pca_train.explained_variance_ratio_))  # eigenvalues

    plt.axvline(x=np.interp(0.9, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)), color='red',
                linestyle='--')
    plt.axhline(y=0.9, color='red', linestyle='--')

    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.savefig('results/explained_variance_chart.png')

    print('Components for 0.8 = ', np.interp(0.8, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)))
    print('Components for 0.85 = ',
          np.interp(0.85, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)))
    print('Components for 0.9 = ', np.interp(0.9, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)))
    print('Components for 0.95 = ',
          np.interp(0.95, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)))
    print('Components for 0.99 = ',
          np.interp(0.99, np.cumsum(pca_train.explained_variance_ratio_), range(n_components)))


def flatten_dataset(dataset):
    flatten_dataset = np.reshape(dataset, (np.shape(dataset)[0],
                                                        np.shape(dataset)[1] *
                                                        np.shape(dataset)[2] *
                                                        np.shape(dataset)[3]))

    # print("X shape:", flatten_dataset.shape)
    # print("X type: ", flatten_dataset[0].dtype)
    # print("X min/max:", np.min(flatten_dataset[0]), np.max(flatten_dataset[0]))

    return flatten_dataset


# #-----------------------------------------------------------------------------
# # Apply PCA
# #-----------------------------------------------------------------------------
#
# n_components = 153
#
# pca = PCA(n_components=n_components)
#
# # fit for train
# pca.fit(X_train)
#
# # Transformation matrices
# cols = 10
# rows = int(n_components/cols)
# fig, axes = plt.subplots(rows, cols, figsize=(20, 70),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
#
# for i, ax in enumerate(axes.flat):
#     ax.imshow(rgb2gray(pca.components_[i].reshape([img_height, img_width, 3])), cmap='bone') #eigenvectors
#     ax.title.set_text('Component ' + str(i))
# plt.savefig('eigenvectors.png')
#
# # transform for both
# images_pca_train_reduced = pca.transform(X_train)
# images_pca_train_recovered = pca.inverse_transform(images_pca_train_reduced)
#
# images_pca_test_reduced = pca.transform(X_test)
# images_pca_test_recovered = pca.inverse_transform(images_pca_test_reduced)
#
# fig, ax = plt.subplots(2, 10, figsize=(20, 5),
#                        subplot_kw={'xticks': [], 'yticks': []},
#                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
#
# # print(np.max(images_pca_train_recovered[1,:].astype("uint8")))
# # print(np.min(images_pca_train_recovered[1,:].astype("uint8")))
# for i in range(0, 10):
#     ax[0, i].imshow(X_train[i].reshape([img_height, img_width, 3]), cmap='gray')
#
#     image_pca_train = images_pca_train_recovered[i, :].reshape([img_height, img_width, 3])
#     ax[1, i].imshow(image_pca_train, cmap='gray')
#
# ax[0, 0].set_ylabel('original')
# ax[1, 0].set_ylabel('reconstruction');
# plt.savefig('image_pca_' + str(n_components) + '.png')
#
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize = (10,6))
# c_map = plt.cm.get_cmap('jet', 5)
# scatter = plt.scatter(images_pca_train_reduced[:, 0], images_pca_train_reduced[:, 1], s = 15,
#             cmap = c_map, c=y_train)
# plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4], label='Class ID')
# plt.xlabel('PC-1') , plt.ylabel('PC-2')
# # plt.show()
# plt.savefig('pc12.png')

def fit_pca(n_components, dataset_flatten):
    pca = PCA(n_components=n_components)

    pca.fit(dataset_flatten)

    print('PCA fit completed')
    return pca


def trans_pca(pca, dataset_train_images_flatten, dataset_test_images_flatten):
    images_pca_train_reduced = pca.transform(dataset_train_images_flatten)
    images_pca_train_recovered = pca.inverse_transform(images_pca_train_reduced)

    images_pca_test_reduced = pca.transform(dataset_test_images_flatten)
    images_pca_test_recovered = pca.inverse_transform(images_pca_test_reduced)

    print('PCA transform completed')
    return images_pca_train_reduced, images_pca_train_recovered, images_pca_test_reduced, images_pca_test_recovered


def show_first_two_pca_components(dataset_train_reduced, y_train):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize = (10,6))
    c_map = plt.cm.get_cmap('jet', 120)
    scatter = plt.scatter(dataset_train_reduced[:, 0], dataset_train_reduced[:, 1], s = 10,
                cmap = c_map, c=y_train)
    plt.colorbar(scatter, ticks=range(0, 120), label='Class ID')
    plt.xlabel('PC-1') , plt.ylabel('PC-2')
    # plt.show()
    plt.savefig('results/pc12.png')

def show_recovered_images(dataset_train_reduced, dataset_train_recovered, n_components, img_height, img_width):
    fig, ax = plt.subplots(2, 10, figsize=(20, 5),
                           subplot_kw={'xticks': [], 'yticks': []},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i in range(0, 10):
        ax[0, i].imshow(dataset_train_reduced[i].reshape([img_height, img_width, 3]), cmap='gray')

        image_pca_train = dataset_train_recovered[i, :].reshape([img_height, img_width, 3])
        ax[1, i].imshow(image_pca_train, cmap='gray')

    ax[0, 0].set_ylabel('original')
    ax[1, 0].set_ylabel('reconstruction')
    plt.savefig('results/image_pca_' + str(n_components) + '.png')
