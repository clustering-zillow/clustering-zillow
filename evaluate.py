import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.formula.api import ols


def plot_residuals(y, yhat, df):
    residual = yhat - y 
    plt.hlines(0, y.min(), y.max(), ls=':')
    plt.scatter(y, residual)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    

def regression_errors(y, yhat, df):
    SSE = mean_squared_error(df.y, df.yhat)*len(df)
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = SSE+ESS
    MSE = mean_squared_error(df.y, df.yhat)
    RMSE = sqrt(mean_squared_error(df.y, df.yhat))
    return {'SSE' : SSE, 'ESS' : ESS, 'TSS' :TSS, 'MSE' : MSE, 'RMSE' : RMSE}

def baseline_mean_errors(y, df):
    df['yhat_bl'] = df.y.mean()
    SSE_bl = mean_squared_error(df.y, df.yhat_bl)*len(df)
    MSE_bl = mean_squared_error(df.y, df.yhat_bl)
    RMSE_bl = sqrt(mean_squared_error(df.y, df.yhat_bl))
    return {'SSE_bl' : SSE_bl, 'MSE_bl' : MSE_bl, 'RMSE_bl' : RMSE_bl}

def better_than_baseline(y, yhat, df):
    m = regression_errors(y, yhat, df)
    b = baseline_mean_errors(y, df)
    if m['RMSE'] < b['RMSE_bl']:
        return True
    else: False
    
    
def model_significance(ols_model):
    r2 = ols_model.rsquared
    print('R-squared = ', round(r2,3))
    f_pval = ols_model.f_pvalue
    print("p-value for model significance = ", round(f_pval,4))