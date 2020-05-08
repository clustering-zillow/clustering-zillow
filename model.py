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
    return RMSE, R2