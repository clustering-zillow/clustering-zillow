import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from statsmodels.formula.api import ols
from math import sqrt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt





def encode(train, validate, test, col_name):

    encoded_values = sorted(list(train[col_name].unique()))

    # Integer Encoding
    int_encoder = LabelEncoder()
    train.encoded = int_encoder.fit_transform(train[col_name])
    validate.encoded = int_encoder.transform(validate[col_name])
    test.encoded = int_encoder.transform(test[col_name])

    # create 2D np arrays of the encoded variable (in train and test)
    train_array = np.array(train.encoded).reshape(len(train.encoded),1)
    validate_array = np.array(validate.encoded).reshape(len(validate.encoded),1)
    test_array = np.array(test.encoded).reshape(len(test.encoded),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    validate_ohe = ohe.fit_transform(validate_array)
    test_ohe = ohe.transform(test_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of train/test
    # then merge the new dataframe with the existing train/test dataframe
    train_encoded = pd.DataFrame(data=train_ohe,
                            columns=encoded_values, index=train.index)
    train = train.join(train_encoded)
    
    test_encoded = pd.DataFrame(data=test_ohe,
                               columns=encoded_values, index=test.index)
    test = test.join(test_encoded)
    
    validate_encoded = pd.DataFrame(data=test_ohe,
                               columns=encoded_values, index=validate.index)
    validate = validate.join(validate_encoded)

    return train, validate, test

def select_rfe(X, y, k):
    lm = LinearRegression()
    rfe = RFE(lm, k)
    X_rfe = rfe.fit_transform(X, y)
    rfe_features = X.loc[:,rfe.support_].columns.tolist()
    print(rfe_features)

def RMSE(y, yhat):
    MSE = mean_squared_error(y, yhat)
    return sqrt(MSE)

def poly_regression(df, feature_list, n ):
    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    poly = PolynomialFeatures(degree=n)
    X_poly = poly.fit_transform(X_scaled)
    lm_poly = LinearRegression()
    lm_poly.fit(X_poly, y_train)
    y_train['predicted_poly'] = lm_poly.predict(X_poly)
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_train.logerror, y_train.predicted_poly))))
    R2 = float('{:.3f}'.format(r2_score(y_train.logerror, y_train.predicted_poly)))
    return RMSE, R2, y_train, lm_poly, poly

def poly_regression_1(df, lm_poly, poly, feature_list ):
    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    X_poly = poly.transform(X_scaled)
    y_train['predicted_poly'] = lm_poly.predict(X_poly)
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_train.logerror, y_train.predicted_poly))))
    R2 = float('{:.3f}'.format(r2_score(y_train.logerror, y_train.predicted_poly)))
    return RMSE, R2, y_train

def linear_reg(df, feature_list):

    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    lm = LinearRegression()
    lm.fit(X_scaled, y_train)
    y_train['predicted'] = lm.predict(X_scaled)
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_train.logerror, y_train.predicted))))
    R2 = float('{:.3f}'.format(r2_score(y_train.logerror, y_train.predicted)))
    return RMSE, R2, lm, y_train

def linear_reg1(df, feature_list, lm):

    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    y_train['predicted'] = lm.predict(X_scaled)
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_train.logerror, y_train.predicted))))
    R2 = float('{:.3f}'.format(r2_score(y_train.logerror, y_train.predicted)))
    return RMSE, R2, y_train

def GAM_model(df, feature_list):
    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    X_scaled= X_scaled.to_numpy()
    y_train = y_train.to_numpy()
    from pygam import LinearGAM, s, f, te
    gam = LinearGAM(s(0) +s(1) +s(2) + s(3) +s(4) +s(5))
    gam.gridsearch(X_scaled,y_train)
    y_pred = gam.predict(X_scaled)
    y_pred = pd.DataFrame(y_pred)
    y_pred['actual'] =y_train
    y_pred.columns = ['predicted', 'actual']
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_pred.actual, y_pred.predicted))))
    R2 = float('{:.3f}'.format(r2_score(y_pred.actual, y_pred.predicted)))
    return RMSE, R2, gam

def GAM_model1(df, feature_list, gam):
    X_train = df[feature_list]
    y_train = df[['logerror']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    X_scaled= X_scaled.to_numpy()
    y_train = y_train.to_numpy()
    y_pred = gam.predict(X_scaled)
    y_pred = pd.DataFrame(y_pred)
    y_pred['actual'] =y_train
    y_pred.columns = ['predicted', 'actual']
    RMSE = float('{:.3f}'.format(sqrt(mean_squared_error(y_pred.actual, y_pred.predicted))))
    R2 = float('{:.3f}'.format(r2_score(y_pred.actual, y_pred.predicted)))
    return RMSE, R2


def run_all_functions_train(df, features):
    index = ['Baseline', 'Linear_model', 'polynomial_LR', 'GAM']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df['base_logerror'] = df.logerror.mean()
    df1.loc['Baseline', 'RMSE'] = sqrt(mean_squared_error(df.logerror,df.base_logerror))
    df1.loc['Baseline', 'R2'] = sqrt(r2_score(df.logerror,df.base_logerror))
    RMSE, R2, lm, y_train = linear_reg(df,features)
    df1.loc['Linear_model', 'RMSE'] = RMSE
    df1.loc['Linear_model', 'R2'] = R2
    RMSE, R2, y_train, lm_poly, poly = poly_regression(df,features, 6)
    df1.loc['polynomial_LR', 'RMSE'] = RMSE
    df1.loc['polynomial_LR', 'R2'] = R2
    RMSE, R2, gam = GAM_model(df,features)
    df1.loc['GAM', 'RMSE'] = RMSE
    df1.loc['GAM', 'R2'] = R2
    return df1, lm_poly, poly, lm, gam

def run_all_functions_test(df, train, features):
    df_train, lm_poly, poly, lm, gam = run_all_functions_train(train,features)
    index = ['Baseline', 'Linear_model', 'polynomial_LR', 'GAM']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df['base_logerror'] = df.logerror.mean()
    df1.loc['Baseline', 'RMSE'] = sqrt(mean_squared_error(df.logerror,df.base_logerror))
    df1.loc['Baseline', 'R2'] = sqrt(r2_score(df.logerror,df.base_logerror))
    RMSE, R2, y_train = linear_reg1(df,features,lm)
    df1.loc['Linear_model', 'RMSE'] = RMSE
    df1.loc['Linear_model', 'R2'] = R2
    RMSE, R2, y_train = poly_regression_1(df, lm_poly, poly, features)
    df1.loc['polynomial_LR', 'RMSE'] = RMSE
    df1.loc['polynomial_LR', 'R2'] = R2
    RMSE, R2 = GAM_model1(df,features, gam)
    df1.loc['GAM', 'RMSE'] = RMSE
    df1.loc['GAM', 'R2'] = R2
    return df1

def cluster_poly_model(df, features):
    df_0 = df[df.cluster == 'cluster_0']
    
    index = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'overall']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df_0 = df[df.cluster == 'cluster_0']
    RMSE, R2, y_train0, lm_poly0, poly0 = poly_regression(df_0,features, 6)
    df1.loc['cluster_0', 'RMSE'] = RMSE
    df1.loc['cluster_0', 'R2'] = R2

    df_1 = df[df.cluster == 'cluster_1']
    RMSE, R2, y_train1, lm_poly1, poly1  = poly_regression(df_1,features, 6)
    df1.loc['cluster_1', 'RMSE'] = RMSE
    df1.loc['cluster_1', 'R2'] = R2

    df_2 = df[df.cluster == 'cluster_2']
    RMSE, R2, y_train2, lm_poly2, poly2  = poly_regression(df_2,features, 6)
    df1.loc['cluster_2', 'RMSE'] = RMSE
    df1.loc['cluster_2', 'R2'] = R2

    df_3 = df[df.cluster == 'cluster_3']
    RMSE, R2, y_train3, lm_poly3, poly3  = poly_regression(df_3,features, 6)
    df1.loc['cluster_3', 'RMSE'] = RMSE
    df1.loc['cluster_3', 'R2'] = R2

    y_train_comb= pd.concat([y_train0, y_train1,y_train2,y_train3])
    RMSE_train = sqrt(mean_squared_error(y_train_comb.logerror,y_train_comb.predicted_poly))
    R2_train = r2_score(y_train_comb.logerror,y_train_comb.predicted_poly)

    
    df1.loc['overall', 'RMSE'] = RMSE_train
    df1.loc['overall', 'R2'] = R2_train
    return df1, y_train_comb, poly0, lm_poly0, poly1, lm_poly1, poly2, lm_poly2, poly3, lm_poly3

def cluster_poly_test(df, features,poly0, lm_poly0, poly1, lm_poly1, poly2, lm_poly2, poly3, lm_poly3 ):
    df_0 = df[df.cluster == 'cluster_0']
    
    index = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'overall']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df_0 = df[df.cluster == 'cluster_0']
    RMSE, R2, y_train0 = poly_regression_1(df_0,lm_poly0, poly0, features)
    df1.loc['cluster_0', 'RMSE'] = RMSE
    df1.loc['cluster_0', 'R2'] = R2

    df_1 = df[df.cluster == 'cluster_1']
    RMSE, R2, y_train1  = poly_regression_1(df_1,lm_poly1, poly1, features)
    df1.loc['cluster_1', 'RMSE'] = RMSE
    df1.loc['cluster_1', 'R2'] = R2

    df_2 = df[df.cluster == 'cluster_2']
    RMSE, R2, y_train2  = poly_regression_1(df_2,lm_poly2, poly2, features)
    df1.loc['cluster_2', 'RMSE'] = RMSE
    df1.loc['cluster_2', 'R2'] = R2

    df_3 = df[df.cluster == 'cluster_3']
    RMSE, R2, y_train3  = poly_regression_1(df_3,lm_poly3, poly3, features)
    df1.loc['cluster_3', 'RMSE'] = RMSE
    df1.loc['cluster_3', 'R2'] = R2

    y_train_comb= pd.concat([y_train0, y_train1,y_train2,y_train3])
    RMSE_train = sqrt(mean_squared_error(y_train_comb.logerror,y_train_comb.predicted_poly))
    R2_train = r2_score(y_train_comb.logerror,y_train_comb.predicted_poly)

    
    df1.loc['overall', 'RMSE'] = RMSE_train
    df1.loc['overall', 'R2'] = R2_train
    return df1, y_train_comb

def clusters(df, kmeans):
    X = df[['finishedsquarefeet12']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    return df


def plot_scatter(x,y):
    f, (ax1) = plt.subplots(1, 1)
    plt.figure(figsize = (12,8))
    ax1.scatter(x, y)
    ax1.plot([-0.5, 0.5], [-0.5, 0.5], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Target vs Predicted')
    ax1.text(-0.4, 0.4, r'$R^2$=%.2f, RMSE=%.3f' % (r2_score(x, y), sqrt(mean_squared_error(x, y))))
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.5, 0.5])
    plt.show()

def cluster_lm_model(df, features):
    df_0 = df[df.cluster == 'cluster_0']
    
    index = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'overall']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df_0 = df[df.cluster == 'cluster_0']
    RMSE, R2, lm0, y_train0 = linear_reg(df_0, features)
    df1.loc['cluster_0', 'RMSE'] = RMSE
    df1.loc['cluster_0', 'R2'] = R2

    df_1 = df[df.cluster == 'cluster_1']
    RMSE, R2, lm1, y_train1 = linear_reg(df_1, features)
    df1.loc['cluster_1', 'RMSE'] = RMSE
    df1.loc['cluster_1', 'R2'] = R2

    df_2 = df[df.cluster == 'cluster_2']
    RMSE, R2, lm2, y_train2 = linear_reg(df_2, features)
    df1.loc['cluster_2', 'RMSE'] = RMSE
    df1.loc['cluster_2', 'R2'] = R2

    df_3 = df[df.cluster == 'cluster_3']
    RMSE, R2, lm3, y_train3 = linear_reg(df_3, features)
    df1.loc['cluster_3', 'RMSE'] = RMSE
    df1.loc['cluster_3', 'R2'] = R2

    y_train_comb= pd.concat([y_train0, y_train1,y_train2,y_train3])
    RMSE_train = sqrt(mean_squared_error(y_train_comb.logerror,y_train_comb.predicted))
    R2_train = r2_score(y_train_comb.logerror,y_train_comb.predicted)

    
    df1.loc['overall', 'RMSE'] = RMSE_train
    df1.loc['overall', 'R2'] = R2_train
    return df1, y_train_comb, lm0, lm1, lm2, lm3

def cluster_lm_test(df, features, lm0, lm1, lm2, lm3):
    df_0 = df[df.cluster == 'cluster_0']
    
    index = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'overall']
    columns = [ 'RMSE', 'R2']
    df1 = pd.DataFrame(index=index, columns=columns)
    df1 = df1.fillna(0) # with 0s rather than NaNs
    df_0 = df[df.cluster == 'cluster_0']
    RMSE, R2, y_train0 = linear_reg1(df_0, features, lm0)
    df1.loc['cluster_0', 'RMSE'] = RMSE
    df1.loc['cluster_0', 'R2'] = R2

    df_1 = df[df.cluster == 'cluster_1']
    RMSE, R2, y_train1 = linear_reg1(df_1, features, lm1)
    df1.loc['cluster_1', 'RMSE'] = RMSE
    df1.loc['cluster_1', 'R2'] = R2

    df_2 = df[df.cluster == 'cluster_2']
    RMSE, R2,  y_train2 = linear_reg1(df_2, features, lm2)
    df1.loc['cluster_2', 'RMSE'] = RMSE
    df1.loc['cluster_2', 'R2'] = R2

    df_3 = df[df.cluster == 'cluster_3']
    RMSE, R2, y_train3 = linear_reg1(df_3, features, lm3)
    df1.loc['cluster_3', 'RMSE'] = RMSE
    df1.loc['cluster_3', 'R2'] = R2

    y_train_comb= pd.concat([y_train0, y_train1,y_train2,y_train3])
    RMSE_train = sqrt(mean_squared_error(y_train_comb.logerror,y_train_comb.predicted))
    R2_train = r2_score(y_train_comb.logerror,y_train_comb.predicted)

    
    df1.loc['overall', 'RMSE'] = RMSE_train
    df1.loc['overall', 'R2'] = R2_train
    return df1, y_train_comb