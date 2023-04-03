import numpy as np
import pandas as pd
import impyute as impy
from scipy import stats
from sklearn.impute import KNNImputer

def lower_upper(df):
    df = df.dropna()
    q25 = np.quantile(df, 0.25)
    q75 = np.quantile(df, 0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    return lower, upper

def imp_min(df, impDf):
    minValue = impDf.min()
    df = df.fillna(minValue)

    return df

def imp_max(df, impDf):
    maxValue = impDf.max()
    df = df.fillna(maxValue)

    return df

def imp_mean(df, impDf):
    meanValue = impDf.mean()
    df = df.fillna(meanValue)

    return df

def imp_median(df, impDf):
    medianValue = impDf.median()
    df = df.fillna(medianValue)

    return df