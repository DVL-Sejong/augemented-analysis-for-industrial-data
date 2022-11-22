import numpy as np
import pandas as pd
import module.imputation as imputation

fileName = 'result_10bike'
originDf = pd.read_csv(fileName + '.csv')
totalValue = len(originDf) * len(list(originDf.columns))

missing = sum(originDf.isnull().sum().values.tolist())
print(100 - ((missing/totalValue) * 100))

tmpList = []
for column in originDf:
    df = pd.DataFrame(pd.to_numeric(originDf[column], errors = 'coerce'))
    df = df.dropna()

    lower, upper = imputation.LowerUpper(df[column])
    data1 = df[df[column] > upper]
    data2 = df[df[column] < lower]
    tmpList.append(data1.shape[0] + data2.shape[0])
outlier = sum(tmpList)
print(100 - ((outlier/totalValue) * 100))

tmpList = []
for column in originDf:
    df = originDf[column].dropna()
    df = pd.DataFrame(pd.to_numeric(df, errors = 'coerce'))
    tmpList.append(df.isnull().sum().values[0].tolist())
incons = sum(tmpList)
print(100 - ((incons/totalValue) * 100))