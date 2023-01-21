from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score


def knn_classifier(dataset_train_features, dataset_train_labels, dataset_test_features, dataset_test_labels,
                   class_names, optimal_k, no_clusters):
    """
         The function make classification ang display results

         @param dataset_train_features:
         @param dataset_train_labels:
         @param dataset_test_features:
         @param dataset_test_labels:
         @param class_names:
         @param optimal_k:
         @param no_clusters:
    """
    classifier = KNeighborsClassifier(
        n_neighbors=optimal_k,
        # metric='minkowski',
        # p=2,
        # algorithm='auto',
        # weights='distance'
    )
    classifier.fit(dataset_train_features, dataset_train_labels)

    y_pred = classifier.predict(dataset_test_features)

    print('\n\nAccuracy: ', accuracy_score(dataset_test_labels, y_pred) * 100, '%')

    # # plot only a part of CM
    if len(class_names) > 40:
        top_classes = [label[0] for label in Counter(dataset_test_labels).most_common()[:40]] + ["_other"]
        dataset_test_labels = [y if y in top_classes else "_other" for y in dataset_test_labels]
        y_pred = [y if y in top_classes else "_other" for y in y_pred]
        class_names = [class_names[index] for index in top_classes if index != '_other'] + ['_other']

    cmx = confusion_matrix(dataset_test_labels, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle('Confusion matrix', fontsize=16)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/confusion-matrix' + str(no_clusters) + '.png')
    plt.clf()


def get_optimal_k_for_knn(image_features, train_labels, no_clusters):
    """
         The function find optimal k

         @param image_features:
         @param train_labels:
         @param no_clusters:
         @return:
            optimal_k
    """
    neighbors = [x for x in range(0, 50+1) if x%2 != 0]
    cross_validation_scores = []

    for k in neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_model, image_features, train_labels, cv=5)
        cross_validation_scores.append(scores.mean())
        print("Accuracy {}".format(k), scores)
    print("\nMaximum accuracy gained in training is:%.3f" % (max(cross_validation_scores) * 100))
    print("\nMinimum Error gained in training is:%.3f" % (1 - (max(cross_validation_scores))))

    # changing to classification error
    error = [1 - x for x in cross_validation_scores]

    # determining best k
    optimal_k = neighbors[error.index(min(error))]
    print("\nThe optimal no. of neighbors is {}".format(optimal_k))

    # plot misclassification error vs k
    plt.figure()
    plt.plot(neighbors, error)
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Error")
    # plt.show()
    plt.savefig('results/error-optimal-k' + str(no_clusters) + '.png')
    plt.clf()

    return optimal_k
