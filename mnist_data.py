import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def q12(X, y):

    ones = 0
    zeros = 0

    one_flag = True
    zero_flag = True

    index = 0

    while index < 10 :

        if y[index] == 0 and zero_flag:
            zeros += 1
            plt.imshow(X[index])
            plt.show()

        if y[index] == 1 and one_flag:
            ones += 1
            plt.imshow(X[index])
            plt.show()

        if zeros == 3:
            zero_flag = False

        if ones == 3:
            one_flag = False

        if not one_flag and not zero_flag:
            break




#Q13

def rearrange_data(X):

    return X.reshape(X.shape[0], 28 * 28)


def compare14():

    results_l = list()
    results_svm = list()
    results_Kneighbors = list()
    results_tree = list()

    logistic = LinearRegression()
    svm = SVC(C=1e10, kernel='linear')
    Kneighbors = KNeighborsClassifier(n_neighbors=2)
    tree = DecisionTreeClassifier(max_depth=10)

    for m in [5, 10, 15, 25, 70]:

        pos_log = list()
        pos_svm = list()
        pos_Kneighbors = list()
        pos_tree = list()

        for i in range(50):

            while True:
                X = np.random.choice(range(y_train[0], m))
                y = y_train[X]

                if 0 not in y or 1 not in y:
                    continue
                else:break

            svm.fit(rearrange_data(x_train[X]), y)
            logistic.fit(rearrange_data(x_train[X]), y)
            Kneighbors.fit(rearrange_data(x_train[X]), y)
            tree.fit(rearrange_data(x_train[X]), y)

            pos_log.append(np.array(logistic.score(rearrange_data(x_test), y_test)))
            pos_svm.append(np.array(svm.score(rearrange_data(x_test), y_test)))
            pos_Kneighbors.append(np.array(Kneighbors.score(rearrange_data(x_test), y_test)))
            pos_tree.append(np.array(tree.score(rearrange_data(x_test), y_test)))

        results_l.append(np.array(pos_log)/50)
        results_svm.append(np.array(pos_svm)/50)
        results_Kneighbors.append(np.array(pos_Kneighbors)/50)
        results_tree.append(np.array(pos_tree)/50)

        plt.plot(m, results_l, label="Mean Accuray of Logistic Alg")
        plt.plot(m, results_svm, label="Mean Accuray of Soft-SVM Alg")
        plt.plot(m, results_Kneighbors, label="Mean Accuray of Nearest Neighbors Alg")
        plt.plot(m, results_tree, label="Mean Accuray of Decision Tree Alg")
        plt.title("Mean accuracy as function of m (SVM, logistic, Kneighbors and Tree) ")
        plt.legend()
        plt.show()