import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans



def plot_variable_pairs(df):
    """Takes a DataFrame and all of the pairwise relationships along with the regression line for each pair"""
    sns.pairplot(df, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.9}})
    plt.show();

def months_to_years(tenure_months, df):
    """returns the dataframe with a new feature tenure_years"""
    df['tenure_years'] = (tenure_months//12)
    return df

def plot_categorical_and_continous_vars(categorical_var, continuous_var, df):
    """outputs 3 different plots for plotting a categorical variable with a continuous variable"""
    #f, axes = plt.subplots(1, sharey=True, figsize=(6, 4))
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.boxplot(x= categorical_var, y= continuous_var, data=df)
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.swarmplot(x= categorical_var, y= continuous_var, data=df)
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.barplot(x= categorical_var, y= continuous_var, data=df)

def plot_scatter(x,y):
    f, (ax1) = plt.subplots(1, 1)

    ax1.scatter(x, y)
    ax1.plot([0, 6000000], [0, 6000000], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Target vs Predicted')
    ax1.text(1, 4000000, r'$R^2$=%.2f, MAE=%.2f' % (r2_score(x, y), median_absolute_error(x, y)))
    ax1.set_xlim([0, 6000000])
    ax1.set_ylim([0, 6000000])
    plt.show()

def plot_scatter_log(x,y):
    f, (ax1) = plt.subplots(1, 1)

    ax1.scatter(x, y)
    ax1.plot([0, 40], [0, 40], '--k')
    ax1.set_ylabel('log2 (Target predicted)')
    ax1.set_xlabel('log2 (True Target)')
    ax1.set_title('Target vs Predicted')
    ax1.text(1, 30, r'$R^2$=%.2f, MAE=%.2f' % (r2_score(x, y), median_absolute_error(x, y)))
    ax1.set_xlim([0, 40])
    ax1.set_ylim([0, 40])
    plt.show()


def cluster(train, X, k):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 539)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    train['cluster'] = kmeans.predict(X_scaled)
    train['cluster'] = 'cluster_' + train.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return train, X_scaled, scaler, kmeans, centroids

def scatter_plot(x,y,train,kmeans, X_scaled, scaler):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = train, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')

