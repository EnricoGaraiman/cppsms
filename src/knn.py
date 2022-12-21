from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def knn_classifier(dataset_train_reduced, dataset_train_labels, dataset_test_reduced, dataset_test_labels, number_classes, class_names):
    classifier = KNeighborsClassifier(n_neighbors=number_classes, metric='minkowski', p=2)
    classifier.fit(dataset_train_reduced, dataset_train_labels)

    y_pred = classifier.predict(dataset_test_reduced)

    cmx = confusion_matrix(dataset_test_labels, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Matrice de confuzie', fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax)
    plt.savefig('results/confusion-matrix.png')


