from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler


def knn_classifier(dataset_train_features, dataset_train_labels, dataset_test_features, dataset_test_labels, class_names):
    print(np.shape(dataset_train_features))
    print(np.shape(dataset_train_labels))
    print(np.shape(dataset_test_features))
    print(np.shape(dataset_test_labels))

    # standardize
    stdslr = StandardScaler().fit(dataset_train_features)
    dataset_train_features = stdslr.transform(dataset_train_features)
    dataset_test_features = stdslr.transform(dataset_test_features)

    # from sklearn.svm import LinearSVC
    # classifier = LinearSVC(max_iter=80000)
    # classifier.fit(dataset_train_features, dataset_train_labels)
    # LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    #           intercept_scaling=1, loss='squared_hinge', max_iter=80000,
    #           multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #           verbose=0)

    for k in range(3,4):
        print(k)
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            metric='minkowski',
            p=2,
            algorithm='brute',
            weights='uniform'
        )
        classifier.fit(dataset_train_features, dataset_train_labels)

        y_pred = classifier.predict(dataset_test_features)

        print('Accuracy: ', accuracy_score(dataset_test_labels, y_pred) * 100, '%')

    # # plot only a part of CM
    # top_classes = [label[0] for label in Counter(dataset_test_labels).most_common()[:40]] + ["_other"]
    # dataset_test_labels = [y if y in top_classes else "_other" for y in dataset_test_labels]
    # y_pred = [y if y in top_classes else "_other" for y in y_pred]
    # class_names = [class_names[index] for index in top_classes if index != '_other'] + ['_other']

    cmx = confusion_matrix(dataset_test_labels, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(30, 30))
    fig.suptitle('Matrice de confuzie', fontsize=16)
    plt.xlabel('True label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.show()
    plt.savefig('results/confusion-matrix.png')


