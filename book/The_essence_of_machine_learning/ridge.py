import numpy as np
from scipy import linalg


class RidgeRegression:
    def __init__(self, lambda_ = 1.):
        self.w_ = None
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, t: np.ndarray):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        c = np.eye(Xtil.shape[1])
        A = np.dot(Xtil.T, Xtil) + self.lambda_ + c
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)

    def predict(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)


if __name__ == '__main__':
    x = np.array([1,2,4,6,7])
    y = np.array([1,3,3,5,4])
    model = RidgeRegression(1.)
    model.fit(x, y)
    b, a = model.w_
