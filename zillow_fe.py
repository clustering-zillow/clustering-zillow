import numpy as np
import pandas as pd

def calculated_features(df):
    # Age = yearbuilt - 2017
    df['age'] = (2017 - df.yearbuilt).astype('int')

    # Tax Rate = taxamount / value
    df['tax_rate'] = ((df.taxamount / df.taxvaluedollarcnt).astype('float').round(4))

    # $/sqft
    df['cost_land_sf'] = ((df.landtaxvaluedollarcnt / df.lotsizesquarefeet).astype('float').round(4))
    df['cost_structure_sf'] = (df.structuretaxvaluedollarcnt / df.finishedsquarefeet12.astype('float').round(4))

    # Has features
    df['has_ac'] = np.where(df.airconditioningtypeid > 0, 1, 0)
    df['has_heating'] = np.where(df.heatingorsystemtypeid == 13, 0, 1)
    df['has_basement'] = np.where(df.basementsqft > 0, 1, 0)
    df['has_fire'] = np.where(df.fireplacecnt > 0, 1, 0)
    df['has_garage'] = np.where(df.garagecarcnt > 0, 1, 0)
    
    return df


def bin_features(df):
    df['sf_bin'] = pd.cut(df.finishedsquarefeet12, 
                              bins = [0, 600, 850, 1050, 1200, 1400, 1600, 1800, 2000, 2200, 2500],
                              labels = [0, 600, 850, 1050, 1200, 1400, 1600, 1800, 2000, 2200])
    df['lotsf_bin'] = pd.cut(df.lotsizesquarefeet, 
                              bins = [0, 3000, 5000, 6000, 7000, 8000, 10000, 15000, 25000, 40000, 50000],
                              labels = [0, 3000, 5000, 6000, 7000, 8000, 10000, 15000, 25000, 40000])
    df['age_bin'] = pd.cut(df.age, 
                              bins = [0, 10, 20, 30, 40, 50, 60, 65, 70, 80, 90],
                              labels = [0, 10, 20, 30, 40, 50, 60, 65, 70, 80])
    df['tax_bin'] = pd.cut(df.tax_rate, 
                              bins = [0, 0.01, 0.011, 0.0114, 0.0118, 0.0121, 0.0125, 0.0132, 0.014, 0.015, 0.02],
                              labels = [0, 0.01, 0.011, 0.0114, 0.0118, 0.0121, 0.0125, 0.0132, 0.014, 0.015])
    return df

