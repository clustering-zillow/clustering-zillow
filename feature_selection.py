import pandas as pd
import numpy as np
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import seaborn as sns
import sklearn

def select_kbest(X, y, n):
    f_selector = sklearn.feature_selection.SelectKBest(f_regression, k = n)
    f_selector.fit(X, y)
    f_support = f_selector.get_support()
    #f_feature = X.loc[:,f_support].columns.tolist()
    f_feature = X.columns[f_support]
    return f_feature   

def rfe(X, y, n):
    lm = LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n)
    X_rfe = rfe.fit(X,y)
    mask = rfe.support_
    rfe_features = X.columns[mask]
    return rfe_features


