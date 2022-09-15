# -*- coding: utf-8 -*-
"""
@author: Timothy Mulumba
"""

import numpy as np, collections, unittest


class OlsEstimator:
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, intercept: bool = True):
        """ 
        OLS Estimator class for obtaining linear regression estimates via Ordinary \
        Least Squares.
        Arguments:
        :param X: numpy nxk-dimensional array of independent variables (matrix or vector)
        :param Y: numpy nx1-dimensional array of dependent variable (vector)
        :param intercept: flag for intercept inclusion
        """
        self.X = X
        self.Y = Y
        self.intercept = intercept
        self.parameters = collections.defaultdict()
        print("OLS Estimator Created")
        
    def fit(self):
        """
        Utility function for parameter estimation.
        
        Arguments:
        
        Returns:
        :beta: Parameter estimate(s)
        :variance_covariance_matrix (VCV): Variance-Covariance of beta
        :residuals: Model residuals
        :Y_hat: Model estimate for Y
        """
        
        assert type(self.X) == np.ndarray and type(self.Y) == np.ndarray, \
        "numpy arrays not used"
        if self.intercept:
            self.X = np.column_stack((np.ones(np.shape(self.X)[0]), self.X))
        XTX = np.dot(self.X.T, self.X) # dot product of matrix X and its transpose
        XTY = np.dot(self.X.T, self.Y)
        beta = np.dot(np.linalg.inv(XTX), XTY)
        
        Y_hat = np.dot(self.X, beta)
        N = np.shape(self.X)[0]
        K = np.shape(beta)[0]
        residuals_variance = np.dot(residuals.T, residuals) / (N-K)
        variance_covariance_matrix = residuals_variance*np.linalg.inv(XTX)
        
        self.parameters['beta'] = beta
        self.parameters['VCV'] = variance_covariance_matrix
        self.parameters['Y_hat'] = Y_hat
        
        return self.parameters
      
    def __str__(self):
        return "An OLS Estimator"    
    
class Test_OLS_mean(unittest.TestCase):
    """
    Class for unit-testing OlsEstimator class
    """
    def setUp(self):
        n_obs, noise_scaling = 1000, 0.1
        self.x = np.random.multivariate_normal(np.zeros(2), np.identity(2), n_obs)
        self.true_betas = np.random.multivariate_normal(np.zeros(2), np.identity(2))
        self.y = self.x @ self.true_betas + noise_scaling*np.random.normal(0, 1, n_obs)
        self.ols = OlsEstimator(self.x, self.y.reshape(-1, 1), intercept=True)
        self.ols.fit()
        

def suite():
    """
    Method for setting up unittest
    """
    s = unittest.TestSuite()
    load_from = unittest.defaultTestLoader.loadTestFromTestCase
    s.addTests(load_from(Test_OLS_mean))
    return s

if __name__ == "__main__":
    t = unittest.TextTestRunner(verbosity=2)
    t.run(suite)

