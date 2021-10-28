import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


#Q7

class Perceptron:

    def __init__(self):

        self.model = [0]

    def fit(self, X, y):

        X = np.array(X)

        XX = np.zeros((X.shape[0] + 1, X.shape[1]))
        XX[1:, :] = X
        XX[0] = 1

        self.model = np.zeros(X.shape[0])

        while True:
            my_arr = (y * np.sign(np.dot(self.model, XX))).reshape(XX.shape[1])
            my_index = np.argmax(my_arr <= 0)

            if my_arr[my_index] > 0:
                break

            self.model = self.model +(y[my_index] * XX[:, my_index])



    def predict(self, X):

        my_sign = np.sign(np.inner(x, self.model[:-1]))
        return my_sign if my_sign != 0 else 1


    def score(self, X, y):

        my_dic = dict()

        my_predict = self.predict(X.T) #TODO A VERIFIER LE TRAnSpOSE
        my_dic["Num_samples"] = len(X)

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        tp = np.sum(np.logical_and(my_predict == 1, y == 1))
        tn = np.sum(np.logical_and(my_predict == -1, y == -1))
        fp = p - tp
        fn = n - tn


        my_dic["Accuracy"] = (tp + tn) / (p + n)
        my_dic["Error"] = (fp + fn) / (p + n)
        my_dic["FPR"] = fp / n
        my_dic["TPR"] = tp / p
        my_dic["Precision"] = tp / (tp + fp)
        my_dic["Recall"] = tp / p

        return my_dic


class Logistic:

    def __init__(self):

        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):

        self.model =self.model.fit(X.T, y)

    def predict(self, x):

        return self.model.predict(x.T)

    def score(self, X, y):

        my_dic = dict()
        my_dic["Accuracy"] =  self.model.score(X, y)
        my_predict = self.predict(X.T)
        my_dic["Num_samples"] = len(X)

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        tp = np.sum(np.logical_and(my_predict == 1, y == 1))
        tn = np.sum(np.logical_and(my_predict == -1, y == -1))
        fp = p - tp
        fn = n - tn

        my_dic["Error"] = (fp + fn) / (p + n)
        my_dic["FPR"] = fp / n
        my_dic["TPR"] = tp / p
        my_dic["Precision"] = tp / (tp + fp)
        my_dic["Recall"] = tp / p

        return my_dic




class DecisionTree:

    def __init__(self):

        self.model = DecisionTreeClassifier()

    def fit(self, X, y):

        self.model.fit(X.T, y)

    def predict(self, x):

        return self.model.predict(x.T)

    def score(self, X, y):

        my_dic = dict()
        my_dic["Accuracy"] = self.model.score(X, y)
        my_predict = self.predict(X)
        my_dic["Num_samples"] = len(X)

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        tp = np.sum(np.logical_and(my_predict == 1, y == 1))
        tn = np.sum(np.logical_and(my_predict == -1, y == -1))
        fp = p - tp
        fn = n - tn

        my_dic["Error"] = (fp + fn) / (p + n)
        my_dic["FPR"] = fp / n
        my_dic["TPR"] = tp / p
        my_dic["Precision"] = tp / (tp + fp)
        my_dic["Recall"] = tp / p

        return my_dic



class SVM:

    def __init__(self):
        self.model = SVC(C=1e10, kernel='linear')


    def fit(self, X, y):
        self.model.fit(X.T, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):

        my_dic = dict()
        my_dic["Accuracy"] = self.model.score(X.T, y)
        my_predict = self.predict(X.T)
        my_dic["Num_samples"] = len(X)

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        tp = np.sum(np.logical_and(my_predict == 1, y == 1))
        tn = np.sum(np.logical_and(my_predict == -1, y == -1))
        fp = p - tp
        fn = n - tn

        my_dic["Error"] = (fp + fn) / (p + n)
        my_dic["FPR"] = fp / n
        my_dic["TPR"] = tp / p
        my_dic["Precision"] = tp / (tp + fp)
        my_dic["Recall"] = tp / p

        return my_dic



class LDA:

    def __init__(self):

        self.model = None
        self.cov = None
        self.probs = None
        self.mu = None

    def fit(self, X, y):

        self.cov = np.cov(X)

        probs_pos = len(np.where(y == 1)[0]) / X.shape[1]
        probs_neg = len(np.where(y == -1)[0]) / X.shape[1]
        self.probs = np.array([probs_pos, probs_neg])

        mu_pos = np.mean(X.T[y == 1], axis=0)
        mu_neg = np.mean(X.T[y == -1], axis=0)
        self.mu = np.array([mu_pos, mu_neg])

    def predict(self, x):

        i = 0
        all_deltas = list()

        while i < len(self.mu):

            new_delta = x.T @ np.linalg.pinv(self.cov) @ self.mu[i] - 0.5 * self.mu[i].T @ np.linalg.pinv(self.cov) @ self.mu[i] + np.log(self.probs[i])
            all_deltas.append(new_delta)

        return np.array([1, -1])[np.argmax(np.array(all_deltas), axis = 0)]


    def score(self, X, y):

        my_dic = dict()

        my_predict = self.predict(X)
        my_dic["Num_samples"] = len(X)

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        tp = np.sum(np.logical_and(my_predict == 1, y == 1))
        tn = np.sum(np.logical_and(my_predict == -1, y == -1))
        fp = p - tp
        fn = n - tn

        my_dic["Accuracy"] = (tp + tn) / (p + n)
        my_dic["Error"] = (fp + fn) / (p + n)
        my_dic["FPR"] = fp / n
        my_dic["TPR"] = tp / p
        my_dic["Precision"] = tp / (tp + fp)
        my_dic["Recall"] = tp / p

        return my_dic