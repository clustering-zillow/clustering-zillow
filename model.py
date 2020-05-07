import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from statsmodels.formula.api import ols
from math import sqrt

def select_rfe(X, y, k):
    lm = LinearRegression()
    rfe = RFE(lm, k)
    X_rfe = rfe.fit_transform(X, y)
    rfe_features = X.loc[:,rfe.support_].columns.tolist()
    print(rfe_features)

def RMSE(y, yhat):
    MSE = mean_squared_error(y, yhat)
    return sqrt(MSE)