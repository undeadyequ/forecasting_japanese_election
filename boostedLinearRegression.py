# Import the packages used to load and manipulate the data
import numpy as np # Numpy is a Matlab-like package for array manipulation and linear algebra
import pandas as pd # Pandas is a data-analysis and table-manipulation tool
import urllib.request # Urlib will be used to download the dataset

# Import the function that performs sample splits from scikit-learn
from sklearn.model_selection import train_test_split

# Import package used to make copies of objects
from copy import deepcopy
# Our boosted linear regression (blr) class will implement 3 methods
# (constructor, fit, and predict), as previously seen in scikit-learn


class blr:
    def __init__(self, learning_rate, max_iter, early_stopping):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.early = early_stopping
        self.y_mean = 0
        self.y_std = 1
        self.x_mean = 0
        self.x_std = 1
        self.theta = 0
        self.mses = []

    def fit(self, x_train_0, y_train_0, x_val_0, y_val_0):
        # Make copies of data to avoid over-writing original dataset
        x_train = deepcopy(x_train_0)
        y_train = deepcopy(y_train_0)
        x_val = deepcopy(x_val_0)
        y_val = deepcopy(y_val_0)

        # De-mean the output variable
        self.y_mean = np.mean(y_train)
        y_train -= self.y_mean
        y_val -= self.y_mean

        # Standardize the output variable
        self.y_std = np.std(y_train)
        y_train /= self.y_std
        y_val /= self.y_std

        # De-mean the input variables
        self.x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_train -= self.x_mean
        x_val -= self.x_mean

        # Standardize the input variables
        self.x_std = np.std(x_train, axis=0, keepdims=True)
        x_train /= self.x_std
        x_val /= self.x_std

        # Initialize counters (total boosting iterations and unproductive iterations)
        current_iter = 0
        no_improvement = 0

        # The starting model has all coefficients equal to zero and predicts a constant zero output
        self.theta = np.zeros((x_train.shape[1], 1))
        y_train_pred = 0 * y_train
        y_val_pred = 0 * y_val
        eta = y_train - y_train_pred
        mses = [np.var(y_val - y_val_pred)]  #

        # Boosting iterations
        while no_improvement < self.early and current_iter < self.max_iter:
            current_iter += 1
            corr_coeffs = np.mean(x_train * eta, axis=0)  # Correlations (equal to betas) beteen residual and inputs
            index_best = np.argmax(np.abs(corr_coeffs))  # Choose variable that has maximum correlation with residual
            self.theta[index_best] += self.lr * corr_coeffs[index_best]  # Parameter update
            y_train_pred += self.lr * corr_coeffs[index_best] * x_train[:, [index_best]]  # Prediction update
            eta = y_train - y_train_pred  # Residuals update
            y_val_pred += self.lr * corr_coeffs[index_best] * x_val[:, [index_best]]  # Validation prediction update
            mses.append(np.var(y_val - y_val_pred))  # New validation MSE
            if mses[-1] > np.min(mses[0:-1]):  # Stopping criterion to avoid over-fitting
                no_improvement += 1
            else:
                no_improvement = 0

        # Final output message
        print('Boosting stopped after ' + str(current_iter) + ' iterations')

    def predict(self, x_test_0):
        # Make copies of the data to avoid over-writing original dataset
        x_test = deepcopy(x_test_0)

        # De-mean input variables using means on training sample
        x_test = x_test - self.x_mean

        # Standardize output variables using standard deviations on training sample
        x_test = x_test / self.x_std

        # Return prediction
        return self.y_mean + self.y_std * np.dot(x_test, self.theta)


