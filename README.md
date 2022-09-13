# Ordinary-Least-Squares
A module that performs linear regression using Ordinary Least Squares and returns:
- Parameter estimates
- Variance Covariance matrix
- Model estimate of the dependent variable, y-hat.

Included is a unit test for the module.

Sample usage:

from OLS import OlsEstimator

np.random.seed(0)
    
X = np.random.randn(400,3)

Y = np.random.randn(400,1)

intercept = True

ols_object = OlsEstimator(X, Y, intercept)

result = ols_object.fit()
