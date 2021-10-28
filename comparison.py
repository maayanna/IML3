import numpy as np
import matplotlib.pyplot as plt
import models
from sklearn.svm import SVC

#Q8
def draw_points(m):

    while True:
        X = np.random.multivariate_normal(np.zeros(2), np.identity(2, dtype=float), m)
        y = np.sign(np.dot(np.array([0.3, -0.5]), X.T) + 0.1)

        if -1 in y or 1 in y:
            X = X.T
            pos, neg = np.where(y == 1), np.where(y == -1)
            plt.plot(X[0, pos], X[1, pos], 'bo', color='blue')  # pos labels
            plt.plot(X[0, neg], X[1, neg], 'o',  color='orange')  # neg labels
            return X, y


#Q9
def plot_all():

    a, b, c = -0.1, -0.3, -0.5

    for m in [5, 10, 15, 25, 70]:

    # m = 5

        X, y = draw_points(m)

        XX = np.array(plt.gca().get_xlim())

        y_vals = ( a + b * XX) / c
        plt.plot(XX, y_vals, label = 'True Hyperplane', color = 'green')

        # SVM
        svm = models.SVM()
        svm.fit(X, y)

        w = svm.model.coef_.reshape(svm.model.coef_.shape[1])
        y_svm = ( -w[0] * XX ) - svm.model.intercept_
        y_svm /= w[1]

        plt.plot(XX, y_svm, label='SVM Hyperplane', color='blue')


        # Perceptron
        perceptron = models.Perceptron()
        perceptron.fit(X, y)

        y_perc = ( -perceptron.model[0] * XX ) - perceptron.model[2]
        y_perc /= perceptron.model[1]
        plt.plot(XX, y_vals, label='Perceptron Hyperplane', color='red')


        plt.legend()
        plt.title("Q9 - Comparaison with m = " + str(m))
        plt.savefig("Q9 - " + str(m) + ".pdf")

        plt.show()




def compare():

    results_p = list()
    results_svm = list()
    results_lda = list()

    perceptron = models.Perceptron()
    svm = models.SVM()
    lda = models.LDA()

    for m in [5, 10, 15, 25, 70]:

        pos_p = list()
        pos_svm = list()
        pos_lda = list()

        for i in range(500):

            while True:
                X = np.random.multivariate_normal(np.zeros(2), np.identity(2, dtype=float), m)
                y = np.sign(np.dot(np.array([0.3, -0.5]), X.T) + 0.1)

                if -1 not in y or 1 not in y:
                    continue
                else:break

            svm.fit(X, y)
            perceptron.fit(X, y)
            lda.fit(X, y)

            X = np.random.multivariate_normal(np.zeros(2), np.identity(2, dtype=float), 10000)
            y = np.sign(np.dot(np.array([0.3, -0.5]), X.T) + 0.1)

            pos_p.append(np.array(perceptron.score(X, y)["Accuracy"]))
            pos_svm.append(np.array(svm.score(X, y)["Accuracy"]))
            pos_lda.append(np.array(lda.score(X, y)["Accuracy"]))

        results_p.append(np.array(pos_p)/500) #mean
        results_svm.append(np.array(pos_svm)/500)
        results_lda.append(np.array(pos_lda)/500)

        plt.plot(m, results_p, label="Mean Accuracy of Perceptron")
        plt.plot(m, results_svm, label="Mean Accuracy of SVM")
        plt.plot(m, results_lda, label="Mean Accuracy of LDA")
        plt.title("Mean accuracy as function of m (SVM, Perceptron and LDA) ")
        plt.legend()
        plt.show()
