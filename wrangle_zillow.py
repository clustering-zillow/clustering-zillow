import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def missing_rows(df):
    new_df = pd.DataFrame(df.isna().sum())
    new_df = new_df.rename(columns={0 : 'num_rows_missing'})
    new_df['pct_rows_missing'] = (new_df.num_rows_missing / len(df)).round(4)
    return new_df

def missing_cols(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1).value_counts())
    new_df = new_df.reset_index()
    new_df = new_df.rename(columns={'index':'num_cols_missing', 0:'num_rows'})
    new_df['pct_cols_missing'] = (new_df.num_cols_missing / df.shape[1])*100
    return new_df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def numeric_to_object(df, num_cols):
    """
    Takes in a dataframe and a list of the columns to be transformed. 
    Changes the type of each column in the list to object type.
    """
    for col in num_cols:
        df[col] = df[col].astype('object')
    return df

def numeric_to_int(df, num_cols):
    """
    Takes in a dataframe and a list of the columns to be transformed. 
    Changes the type of each column in the list to integer type.
    """
    for col in num_cols:
        df[col] = df[col].astype('int')
    return df

def wrangle_zillow():
    df = pd.read_csv('zillow.csv', index_col='id')
    # drop extra column that comes in from csv files
    df = df.drop(columns='Unnamed: 0')
    
    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    
    # drop unnecessary columns
    df = remove_columns(df, ['architecturalstyletypeid',
                             'buildingclasstypeid',
                             'finishedsquarefeet13',
                             'finishedsquarefeet15',
                             'finishedsquarefeet50',
                             'finishedsquarefeet6',
                             'finishedfloor1squarefeet',
                             'pooltypeid10',
                             'pooltypeid2',
                             'pooltypeid7',
                             'fireplaceflag',
                             'airconditioningdesc',
                             'storydesc',
                             'heatingorsystemdesc',
                             'architecturalstyledesc',
                             'buildingclassdesc',
                             'typeconstructiondesc',
                             'yardbuildingsqft17',
                             'yardbuildingsqft26',
                             'calculatedbathnbr',
                             'fullbathcnt',
                             'threequarterbathnbr',
                             'typeconstructiontypeid',
                             'storytypeid',
                             'propertyzoningdesc', 
                            'calculatedfinishedsquarefeet', 
                             'regionidneighborhood',
                             'regionidcity',
                             'regionidcounty',
                             'propertylandusetypeid',
                             'rawcensustractandblock',
                             'propertylandusedesc',
                            'assessmentyear'])

    # Replace Y in taxdelinquencyflag with 1
    df.taxdelinquencyflag = np.where(df.taxdelinquencyflag == 'Y', 1, 0)
    
    # fill nulls with 0 
    df.airconditioningtypeid.fillna(0, inplace=True)
    df.basementsqft.fillna(0, inplace=True)
    df.decktypeid.fillna(0, inplace=True)
    df.fireplacecnt.fillna(0, inplace=True)
    df.garagecarcnt.fillna(0, inplace=True)
    df.garagetotalsqft.fillna(0, inplace=True)
    df.hashottuborspa.fillna(0, inplace=True)
    df.lotsizesquarefeet.fillna(0, inplace=True)
    df.poolcnt.fillna(0, inplace=True)
    df.poolsizesum.fillna(0, inplace=True)
    df.taxdelinquencyyear.fillna(0, inplace=True)
    
    # For heating type, None = 13
    df.heatingorsystemtypeid.fillna(13, inplace=True)
    
    # Fill nulls with most common value
    df.numberofstories.fillna(1, inplace=True)
    df.unitcnt.fillna(1, inplace=True)
    df.yearbuilt.fillna(1955, inplace=True)
    
    # This piece bothers me: I'd rather fill in with a range of values based on the other features
    # Do more exploration on this feature
    df.buildingqualitytypeid.fillna(6, inplace=True)
    
    # Drop rows with null values in certain columns
    df = df.dropna(subset=['structuretaxvaluedollarcnt',
                           'taxvaluedollarcnt',
                           'taxamount',
                           'censustractandblock',
                           'regionidzip',
                           'finishedsquarefeet12'])
    
    # set type for each column
    df = numeric_to_object(df, ['fips',
                                'propertycountylandusecode',
                                'regionidcounty',
                                'regionidzip',
                                'censustractandblock'])
    df = numeric_to_int(df, ['airconditioningtypeid',
                             'basementsqft',
                             'bedroomcnt',
                             'buildingqualitytypeid',
                             'decktypeid',
                             'finishedsquarefeet12',
                             'fireplacecnt',
                             'garagecarcnt',
                             'garagetotalsqft',
                             'hashottuborspa',
                             'heatingorsystemtypeid',
                             'lotsizesquarefeet',
                             'poolcnt',
                             'poolsizesum',
                             'roomcnt',
                             'unitcnt',
                             'yearbuilt',
                             'numberofstories',
                             'assessmentyear',
                             'taxdelinquencyflag',
                             'taxdelinquencyyear'])

    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))
    return df
    
def split_my_data(df, train_pct):
    '''
    Takes in df, train_pct and returns 2 items:
    train, test

    When using this function, in order to have usable datasets, be sure to call it thusly:
    train, test = split_my_data(df, train_pct)
    '''
    return train_test_split(df, train_size = train_pct, random_state = 294)

def min_max_scaler(train, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm

    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, test


