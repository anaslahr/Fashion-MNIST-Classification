import matplotlib
import pandas
from statsmodels.sandbox.nonparametric.tests.ex_gam_new import tic

from mnist_reader import *
import sys
import time
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def call_svm(kernel_choice, X_train, y_train, X_test, y_test):
    # Pickle the Classifier for Future Use
    print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
    print('\nPickling the Classifier for Future Use...')
    clf = svm.SVC(0.1, kernel_choice)
    clf.fit(X_train, y_train)
    with open('MNIST_SVM.pickle', 'wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('MNIST_SVM.pickle', 'rb')
    clf = pickle.load(pickle_in)

    print('\nCalculating Accuracy of trained Classifier...')
    acc = clf.score(X_test, y_test)

    print('\nMaking Predictions on Validation Data...')
    y_pred = clf.predict(X_test)

    print('\nCalculating Accuracy of Predictions...')
    accuracy = accuracy_score(y_test, y_pred)

    print('\nCreating Confusion Matrix...')
    conf_mat = confusion_matrix(y_test, y_pred)

    print('\nSVM Trained Classifier Accuracy: ', acc)
    print('\nPredicted Values: ', y_pred)
    print('\nAccuracy of Classifier on Validation Images: ', accuracy)
    print('\nConfusion Matrix: \n', conf_mat)

    # Plot Confusion Matrix Data as a Matrix
    plt.matshow(conf_mat)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print('\nMaking Predictions on Test Input Images...')
    test_labels_pred = clf.predict(X_test)

    print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
    acc = accuracy_score(y_test, test_labels_pred)

    print('\n Creating Confusion Matrix for Test Data...')
    conf_mat_test = confusion_matrix(y_test, test_labels_pred)

    print('\nPredicted Labels for Test Images: ', test_labels_pred)
    print('\nAccuracy of Classifier on Test Images: ', acc)
    print('\nConfusion Matrix for Test Data: \n', conf_mat_test)

    toc = time.time()

    print('Total Time Taken: {0} ms'.format((toc - tic) * 1000))

    # Plot Confusion Matrix for Test Data
    plt.matshow(conf_mat_test)
    plt.title('Confusion Matrix for Test Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # sys.stdout = old_stdout
    # log_file.close()

    arr = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # Show the Test Images with Original and Predicted Labels
    a = np.random.randint(1, 40, 15)
    if X_test.shape[1] > 32:
        for i in a:
            two_d = (np.reshape(X_test[i], (28, 28)) * 255).astype(np.uint8)
            plt.title('Original Label: {0}  Predicted Label: {1}'.format(arr[y_test[i]], arr[test_labels_pred[i]]))
            plt.imshow(two_d, interpolation='nearest')

        plt.show()


if __name__ == '__main__':
    print('\nLoading Training Data...')
    img_train, labels_train = load_mnist('data/', kind='train')
    train_img = np.array(img_train)
    train_labels = np.array(labels_train)

    print('\nLoading Testing Data...')
    img_test, labels_test = load_mnist('data/', kind='t10k')
    test_img = np.array(img_test)
    test_labels = np.array(labels_test)

    X_train = train_img
    y_train = train_labels

    X_test = test_img
    y_test = test_labels
    call_svm('poly', X_train, y_train, X_test, y_test)
# print('\nPreparing Classifier Training and Validation Data...')
