from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from scipy import stats

pathDir = 'bike/'
fileName = '10bike'

# origin dataframe
df1 = pd.read_csv(pathDir + fileName + '.csv')
df1 = df1.reindex(sorted(df1.columns), axis = 1)
df1 = df1.apply(pd.to_numeric, errors = 'coerce')

# data quality improvement dataframe
df2 = pd.read_csv('result_' + fileName + '.csv')
df2 = df2.reindex(sorted(df2.columns), axis = 1)
df2 = df2.apply(pd.to_numeric, errors = 'coerce')

columnList = list(df2.columns)

### skewness, kurtosis - column
df1 = df1['cnt'].dropna()
df2 = df2['cnt'].dropna()

skewness = abs(stats.skew(df2))
kurtosis = abs(stats.kurtosis(df2))
kstest = abs(stats.ks_2samp(df1, df2).pvalue)

print(skewness)
print(kurtosis)
print(kstest)

### skewness, kurtosis - all column
# skewnesstList = []
# kurtosisList = []
# for i in range(len(columnList)):
#   columnDf = df2.iloc[:, i].dropna()

#   skewness = abs(stats.skew(columnDf))
#   kurtosis = abs(stats.kurtosis(columnDf))
#   skewnesstList.append(skewness)
#   kurtosisList.append(kurtosis)

# skewness = sum(skewnesstList)/len(columnList)
# kurtosis = sum(kurtosisList)/len(columnList)
# print(skewness)
# print(kurtosis)

### kstest - all column
# kstestList = []
# for i in range(len(columnList)):
#   columnDf1 = df1.iloc[:, i].dropna()
#   columnDf2 = df2.iloc[:, i].dropna()

#   kstest = abs(stats.ks_2samp(columnDf1, columnDf2).pvalue)
#   kstestList.append(kstest)

# kstest = sum(kstestList)/len(columnList)
# print(kstest)

# column dataframe
# df1 = df1['quality']
# df2 = df2['quality']

# skewness = abs(stats.skew(df2))
# kurtosis = abs(stats.kurtosis(df2))

# print(skewness)
# print(kurtosis)

### TSNE - normal ndarray
# dataMatrix = df1.dropna().values
# tsneList = TSNE(n_components = 1, random_state = 0).fit_transform(dataMatrix)
# tsneDf = pd.DataFrame(tsneList, columns = ['value'])

# mu = tsneDf.mean()
# std = tsneDf.std()
# rv = stats.norm(loc = mu, scale = std)
# normalList = rv.rvs(size = 5000, random_state = 0)

# skewness = abs(stats.skew(normalList))
# kurtosis = abs(stats.kurtosis(normalList))

# print(skewness)
# print(kurtosis)

### TSNE - data ndarray
# dataMatrix = df2.dropna().values
# tsneList = TSNE(n_components = 1, random_state = 0).fit_transform(dataMatrix)
# tsneList = tsneList.flatten()

# skewness = abs(stats.skew(tsneList))
# kurtosis = abs(stats.kurtosis(tsneList))
# kstest = stats.ks_2samp(normalList, tsneList).pvalue

# print(skewness)
# print(kurtosis)
# print(kstest)