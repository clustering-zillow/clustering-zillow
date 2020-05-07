import numpy as np
import pandas as pd

def calculated_features(df):
    # Age = yearbuilt - 2017
    df['age'] = (2017 - df.yearbuilt).astype('int')

    # Tax Rate = taxamount / value
    df['tax_rate'] = (df.taxamount / df.taxvaluedollarcnt).astype('float').round(4)

    # $/sqft
    df['cost_land_sf'] = (df.landtaxvaluedollarcnt / 
                           (df.lotsizesquarefeet + 0.0001))
                          .astype('float').round(4)
    df['cost_structure_sf'] = (df.structuretaxvaluedollarcnt / df.finishedsquarefeet12).astype('float').round(4)

    # Has features
    df['has_ac'] = np.where(df.airconditioningtypeid > 0, 1, 0)
    df['has_heating'] = np.where(df.heatingorsystemtypeid == 13, 0, 1)
    df['has_fire'] = np.where(df.fireplacecnt > 0, 1, 0)
    df['has_garage'] = np.where(df.garagecarcnt > 0, 1, 0)
    df['has_deck'] = np.where(zillow.decktypeid > 0, 1, 0)
    
    # How many features
    df['is_extra'] = (df.airconditioningtypeid +
                      df.has_deck +
                      df.has_fire +
                      df.has_garage +
                      df.hashottuborspa +
                      df.has_heating +
                      df.poolcnt)
    
    # one hot encode counties
    df['LA'] = np.where(df.county == 'Los_Angeles', 1, 0)
    df['OC'] = np.where(df.county == 'Orange', 1, 0)
    df['VC'] = np.where(df.county == 'Ventura', 1, 0)
    
    return df


def bin_features(df):
    df['sf_bin'] = pd.cut(df.finishedsquarefeet12, 
                              bins = [0, 800, 1100, 1400, 1800, 2200, 10000],
                              labels = [0, 800, 1100, 1400, 1800, 2200])
    df['lotsf_bin'] = pd.cut(df.lotsizesquarefeet, 
                              bins = [0, 4000, 6000, 8000, 20000, 40000, 100000],
                              labels = [0, 4000, 6000, 8000, 20000, 40000])
    df['age_bin'] = pd.cut(df.age, 
                              bins = [0, 15, 30, 45, 60, 70, 120],
                              labels = [0, 15, 30, 45, 60, 70])
    df['tax_bin'] = pd.cut(df.tax_rate, 
                              bins = [0, 0.0105, 0.0115, 0.012, 0.0125, 0.015, 0.5],
                              labels = [0, 0.0105, 0.0115, 0.012, 0.0125, 0.015])
    
    return df

def drop_unnecessary_features(df):
    df = df.drop(columns=['yearbuilt',
                          'airconditioningtypeid',
                          'heatingorsystemtypeid',
                          'basementsqft',
                          'fireplacecnt',
                          'decktypeid',
                          'unitcnt',
                          'propertycountylandusecode'])
    return df

def zillow_fe(df):
    df = calculated_features(df)
    df = bin_features(df)
    df = drop_unnecessary_features(df)
    return df