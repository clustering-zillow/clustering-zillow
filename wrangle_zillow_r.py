import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from math import radians, sin, cos, sqrt, asin

def single_units(df):
    df = df[(df.propertylandusedesc=='Single Family Residential') & ((df.unitcnt<=1)|df.unitcnt.isnull()) & (df.bedroomcnt>0)\
           & (df.bathroomcnt>0) & (df.calculatedfinishedsquarefeet>350)]
    return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df



def haversine(lat1, lon1):
    lat2 = 34.724
    lon2 = -117.882
    R = 6372.8  # Earth radius in kilometers
 
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
 
    a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
    c = 2 * asin(sqrt(a))
 
    return R * c

def zillow_impute_city(df):
    df['haversine_distance'] = [haversine(x, y) for x, y in zip(df.latitude/1000000, df.longitude/1000000)]
    knn_imputer = KNNImputer(n_neighbors=1)
    knn_imputer.fit(df[['haversine_distance', 'regionidcity',]])
    c = pd.DataFrame(knn_imputer.transform(df[['haversine_distance', 'regionidcity', ]]), columns = ['haversine', 'regionid_city'], index = df.parcelid)
    df = pd.merge(df, c,left_on= 'parcelid', right_on='parcelid')
    df = df.drop(columns = ['haversine', 'regionidcity'])
    
    return df

def zillow_impute_zip(df):
    df['haversine_distance'] = [haversine(x, y) for x, y in zip(df.latitude/1000000, df.longitude/1000000)]
    knn_imputer = KNNImputer(n_neighbors=1)
    knn_imputer.fit(df[['haversine_distance', 'regionidzip',]])
    c = pd.DataFrame(knn_imputer.transform(df[['haversine_distance', 'regionidzip', ]]), columns = ['haversine', 'regionid_zip'], index = df.parcelid)
    df = pd.merge(df, c,left_on= 'parcelid', right_on='parcelid')
    df = df.drop(columns = ['haversine', 'regionidzip'])
    df['regionid_zip'] = df.regionid_zip.astype(int)
    
    return df

def zillow_impute_census(df):
    df['haversine_distance'] = [haversine(x, y) for x, y in zip(df.latitude/1000000, df.longitude/1000000)]
    knn_imputer = KNNImputer(n_neighbors=1)
    knn_imputer.fit(df[['haversine_distance', 'censustractandblock',]])
    c = pd.DataFrame(knn_imputer.transform(df[['haversine_distance', 'censustractandblock', ]]), columns = ['haversine', 'census_tractandblock'], index = df.parcelid)
    df = pd.merge(df, c,left_on= 'parcelid', right_on='parcelid')
    df = df.drop(columns = ['haversine', 'censustractandblock'])
    df['census_tractandblock'] = df.census_tractandblock.astype(object)
    
    return df


def zillow_impute_knn(df):
    df = zillow_impute_zip(df)
    df = zillow_impute_census(df)
    return df


def wrangle_zillow(df):
    df = single_units(df)
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .70)
    df = zillow_impute(df)
    df['fips']=df[['fips']].replace({6037.0:'Los_Angeles_county',6059.0:'Orange_county', 6111: 'Ventura_county'})
    return df

