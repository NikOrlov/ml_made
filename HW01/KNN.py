import numpy as np


class KNearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt((X[i] - self.X_train[j]) @ (X[i] - self.X_train[j]).T)
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            row_dists = X[i, :] - self.X_train
            dists[i, :] = np.sqrt(np.sum(row_dists * row_dists, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        sub = np.expand_dims(X, axis=1) - np.expand_dims(self.X_train, axis=0)
        dists = np.sqrt(np.sum(sub * sub, axis=2))
        return dists

    def predict(self, X, k=5, num_loops=0):
        if num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        else:
            raise ValueError(f'Invalid value {num_loops} for num_loops')
        return self.predict_labels(dists, k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            top_k_args = np.argsort(dists[i])[:k]
            closest_y = self.y_train[top_k_args]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred
