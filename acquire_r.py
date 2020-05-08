import pandas as pd
import numpy as np
import env
from env import google_key

def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def get_zillow_data():
    query = """
    SELECT prop.*, 
       pred.logerror, 
       pred.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 
    FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
    WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL
    """
    # SELECT * FROM properties_2017
    # RIGHT JOIN `predictions_2017` USING (`parcelid`)
    # LEFT JOIN `architecturalstyletype` USING (`architecturalstyletypeid`) 
    # LEFT JOIN `buildingclasstype` USING (`buildingclasstypeid`)
    # LEFT JOIN `heatingorsystemtype` USING (`heatingorsystemtypeid`)
    # LEFT JOIN `propertylandusetype` USING (`propertylandusetypeid`)
    # LEFT JOIN `storytype` USING (`storytypeid`)
    # LEFT JOIN `airconditioningtype` USING (`airconditioningtypeid`)
    # LEFT JOIN `typeconstructiontype` USING (`typeconstructiontypeid`) 
    # WHERE properties_2017.latitude IS NOT NULL OR  properties_2017.longitude IS NOT NULL
    
    df = pd.read_sql(query, get_db_url('zillow'))
    # # Drop one of the duplicate column named id
    # df = pd.concat([
    # df.iloc[:, :11], # all the rows, and up to, but not including the index of the column to drop
    # df.iloc[:, 11 + 1:] # all the rows, and everything after the column to drop
    # ], axis=1)
    # #sort values by parcelid and transactiondate
    # df = df.sort_values(["parcelid", "transactiondate"])

    # # drop duplicates
    # df = df.drop_duplicates(subset=["parcelid"],keep='last')
    return df

def rows_missing(df):
    df1 = pd.DataFrame(df.isnull().sum(), columns = ['num_rows_missing'])
    df1['pct_rows_missing'] = df1.num_rows_missing/df.shape[0]
    return df1  

def cols_missing(df):
    """function that takes in a dataframe of observations and attributes and returns a dataframe where each row is
    an atttribute name, the first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute"""
    df['total'] = df.apply(lambda row: row.isnull().sum(), axis =1)
    df2 = df[['total']]
    df2 = df2.reset_index()
    df2 = df2.groupby('total').count()
    df2= df2.reset_index()
    df2.rename(columns = {'total': 'num_cols_missing', 'index': 'num_rows'}, inplace = True)
    df2['pct_cols_missing'] = df2.num_cols_missing/df.shape[1]
    return df2

def zip_code(list):
    geocode_results = []
    for i in range (0, len(list)):
        geocode_results.append(gmaps.reverse_geocode(list[i])[1]['formatted_address'])
    return geocode_results

def get_google(df):
    gmaps = googlemaps.Client(key=google_key)
    df1 =  df[['latitude', 'longitude', 'regionidzip']]
    df1 = df1.dropna()
    df1 = df1.drop_duplicates('regionidzip', keep='first')
    df2 = df1.copy()
    df1 = df1.drop(columns = 'regionidzip')
    records = df1.to_records(index=False)
    list2 = list(records)
    new_zip = zip_code(list2)
    new_zip[0].split()[-2][0:5]
    zip_list = []
    for i in range (0, len(new_zip)):
        zip_list.append(new_zip[i].split()[-2][0:5])
    df3 = pd.DataFrame(zip_list)
    df2 = df2.reset_index()
    zip_codes = pd.concat([df2, df3], axis =1)
    zip_codes = zip_codes.set_index('index').rename(columns = {0:'new_zip'})
    return zip_codes



