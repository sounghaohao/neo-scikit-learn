import numpy as np
from scipy import optimize, linalg


class LinearRegression():
    def __init__(self, positive=False):

        self.positive = positive

    def fit(self, X, y):

        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = [optimize.nnls(X, y[:, j]) for j in range(y.shape[1])]
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        return self.coef_
