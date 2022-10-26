import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : float
            single number with MSE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.

        Comment: If Y is two-dimentional, average the error over both dimentions.
        """
        # YOUR CODE HERE
        return np.mean((Y - X @ w) ** 2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return: float
            single number with MAE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        Comment: If Y is two-dimentional, average the error over both dimentions.
        """
        # YOUR CODE HERE
        return np.mean(np.abs(Y - X @ w))

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return: float
            single number with sum of squared elements of the weight matrix ( \sum_{ij} w_{ij}^2 )
        Computes the L2 regularization term for the weight matrix w.
        """
        # YOUR CODE HERE
        return np.sum(w ** 2)

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)
        Return : float
            single number with sum of the absolute values of the weight matrix ( \sum_{ij} |w_{ij}| )

        Computes the L1 regularization term for the weight matrix w.
        """
        # YOUR CODE HERE
        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):
        """
        Simply ignores the regularization
        """
        return 0.

    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`
        Computes the MSE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.

        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """
        # YOUR CODE HERE
        norm_coeff = 1
        for dim in Y.shape:
            norm_coeff *= dim
        norm_matr = np.ones_like(w) * norm_coeff
        Q = -2 * np.dot(X.T, Y) + 2 * np.dot(np.dot(X.T, X), w)
        return Q / norm_matr

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`
        Computes the MAE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.

        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

        # YOUR CODE HERE
        def mae_derivative_positive(X, Y, w):
            Q_w = -X.T @ np.ones_like(Y)
            norm_coeff = 1
            for dim in Y.shape:
                norm_coeff *= dim
            return Q_w / norm_coeff

        Q = Y - X @ w

        y_mask_pos = (Q >= 0) * 1
        x_mask_pos = (np.sum(y_mask_pos, axis=1) > 0) * 1

        y_mask_neg = (Q < 0) * 1
        x_mask_neg = (np.sum(y_mask_neg, axis=1) > 0) * 1

        pos_der = mae_derivative_positive(X * x_mask_pos[:, np.newaxis], Y, w)
        neg_der = mae_derivative_positive(X * x_mask_neg[:, np.newaxis], Y, w)
        return pos_der - neg_der

    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : numpy array of same shape as `w`
        Computes the L2 regularization term derivative w.r.t. the weight matrix w.
        """

        # YOUR CODE HERE

        return 2 * w

    @staticmethod
    def l1_reg_derivative(w):
        """
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : numpy array of same shape as `w`
        Computes the L1 regularization term derivative w.r.t. the weight matrix w.
        """

        # YOUR CODE HERE
        mask_pos = (w > 0) * 1
        mask_neg = (w < 0) * (-1)
        return mask_pos + mask_neg

    @staticmethod
    def no_reg_derivative(w):
        """
        Simply ignores the derivative
        """
        return np.zeros_like(w)