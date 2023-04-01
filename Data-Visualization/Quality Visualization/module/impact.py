import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import module.main as main
from pycaret.regression import *

uploadFileName = 'bike'
targetColumn = 'cnt'
inputModelList = ['lr', 'svm', 'gbr']
inputEvalList = ['MAE', 'MSE', 'RMSE']
totalEvalList = ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE', 'TT (Sec)']

df = pd.read_csv(uploadFileName + '.csv')
df = df.reindex(sorted(df.columns), axis = 1)
columnList = list(df.columns)

issueList = ['m', 'o', 'i', 'd', 'c', 'r']
missingList = ["remove", "min", "max", "mean", "median"]
outlierList = ["iqr", "zscore"]
corrList = ["pearson", "spearman", "kendall"]
corrThreshold = 0.8





df = pd.read_csv(uploadFileName + '.csv')
columnList = list(df.columns)

# incons, missing drop for autoML
df = df.apply(pd.to_numeric, errors = 'coerce')
df = pd.DataFrame(df, columns = columnList)
df = df.dropna()
df = df.reset_index(drop = True)

# autoML
clf = setup(data = df, target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
model = compare_models(include = inputModelList)
resultDf = pull()

for evalMetric in totalEvalList:
    if evalMetric not in inputEvalList:
        resultDf = resultDf.drop([evalMetric], axis = 1)

resultDict = resultDf.to_dict()

with open('dataset/before.json', 'w') as file:
    file.write(json.dumps(resultDict, indent = 4))





actionDfList = []
for issue in issueList:
    print(issue)

    if issue == 'm':
        for action in missingList:
            actionColumnDfList = []

            if action == "remove":
                columnConcatDf = df.dropna()
                columnConcatDf = columnConcatDf.reset_index(drop = True)

            else:
                for column in columnList:
                    columnDf = df.loc[:, [column]]
                    missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

                    if len(missingIndex) > 0:
                        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
                        inconsDropDf = inconsNaNDf.dropna()

                        if action == "min":
                            columnDf = main.imp_min(columnDf, inconsDropDf)
                        if action == "max":
                            columnDf = main.imp_max(columnDf, inconsDropDf)
                        if action == "mean":
                            columnDf = main.imp_mean(columnDf, inconsDropDf)
                        if action == "median":
                            columnDf = main.imp_median(columnDf, inconsDropDf)
                        
                    else:
                        columnDf = columnDf
                    
                    actionColumnDfList.append(columnDf)

                columnConcatDf = actionColumnDfList[0]
                for i in range(len(actionColumnDfList) - 1):
                    columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[i + 1]], axis = 1, join = 'inner')
                    columnConcatDf = columnConcatDf.reset_index(drop = True)

            columnConcatDf = columnConcatDf.reset_index(drop = True)
            actionDfList.append(columnConcatDf)
    
    if issue == 'o':
        actionColumnDfList = []

        for action in outlierList:
            for column in columnList:
                columnDf = df.loc[:, [column]]
                missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

                inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
                missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
                inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

                if action == "iqr":
                    lower, upper = main.lower_upper(inconsNaNDf)
                    lowerIdxList = list(inconsNaNDf[inconsNaNDf[column] > upper].index.values)
                    upperIdxList = list(inconsNaNDf[inconsNaNDf[column] < lower].index.values)
                    outlierIndex = lowerIdxList + upperIdxList

                if action == "zscore":
                    inconsNaNSeries = inconsNaNDf.dropna()
                    meanValue = np.mean(inconsNaNSeries)
                    stdValue = np.std(inconsNaNSeries)
                    
                    outlierIndex = []
                    zscoreThreshold = 3
                    for i in range(len(inconsNaNDf)):
                        value = inconsNaNDf.iloc[i].values[0]
                        zscore = ((value - meanValue)/stdValue).values[0]

                        if zscore > zscoreThreshold:
                            outlierIndex.append(i)
                
                for idx in missingIndex:
                    columnDf.loc[idx, column] = np.nan

                for idx in inconsIndex:
                    columnDf.loc[idx, column] = 'incons'

                columnDf = columnDf.drop(outlierIndex)
                actionColumnDfList.append(columnDf)

            columnConcatDf = actionColumnDfList[0]
            for i in range(len(actionColumnDfList) - 1):
                columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[i + 1]], axis = 1, join = 'inner')
                columnConcatDf = columnConcatDf.reset_index(drop = True)

            columnConcatDf = columnConcatDf.reset_index(drop = True)
            actionDfList.append(columnConcatDf)

    if issue == 'i':
        actionColumnDfList = []

        for column in columnList:
            columnDf = df.loc[:, [column]]
            missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]
            
            inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
            missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
            inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

            columnDf = columnDf.drop(inconsIndex)
            actionColumnDfList.append(columnDf)

        columnConcatDf = actionColumnDfList[0]
        for i in range(len(actionColumnDfList) - 1):
            columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[i + 1]], axis = 1, join = 'inner')
            columnConcatDf = columnConcatDf.reset_index(drop = True)

        columnConcatDf = columnConcatDf.reset_index(drop = True)
        actionDfList.append(columnConcatDf)
    
    if issue == 'd':
        df = df.drop_duplicates()
        df = df.reset_index(drop = True)
        actionDfList.append(df)

    if issue == 'c' or issue == 'r':
        for action in corrList:
            inconsNaNSeries = df.apply(pd.to_numeric, errors = 'coerce')
            inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
            allCorrDf = inconsNaNDf.corr(method = action)

            highCorrList = []
            if issue == 'c':
                for row in columnList:
                    for column in columnList:
                        if row == column: break
                        if allCorrDf.loc[row][column] > corrThreshold:
                            highCorrList.append([row, column])

                highCorrList = list(set(sum(highCorrList, [])))
                if targetColumn in highCorrList: highCorrList.remove(targetColumn)
                df = df.drop(highCorrList, axis = 1)

            if issue == 'r':
                columnCorrDf = allCorrDf[targetColumn]
                
                for row in columnList:
                    if columnCorrDf[row] > corrThreshold:
                        highCorrList.append(row)

                if targetColumn in highCorrList: highCorrList.remove(targetColumn)
                df = df.drop(highCorrList, axis = 1)

            actionDfList.append(df)

print(len(actionDfList))

# incons, missing drop for autoML
dfList = []
for i in range(len(actionDfList)):
    columnList = list(actionDfList[i].columns)

    df = actionDfList[i].apply(pd.to_numeric, errors = 'coerce')
    df = pd.DataFrame(df, columns = columnList)
    df = df.dropna()
    df = df.reset_index(drop = True)

    dfList.append(df)

for i in range(len(dfList)):
    df = dfList[i]
    df.to_csv('dataset/' + str(i) + '.csv', index = False)

resultList = []
for i in range(0, 15):
    df = pd.read_csv('dataset/' + str(i) + '.csv')

    # autoML
    clf = setup(data = df, target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
    model = compare_models(include = inputModelList)
    resultDf = pull()

    for evalMetric in totalEvalList:
        if evalMetric not in inputEvalList:
            resultDf = resultDf.drop([evalMetric], axis = 1)

    resultDict = resultDf.to_dict()
    resultList.append(resultDict)

# autoML result to file
with open('dataset/after.json', 'w') as file:
    file.write(json.dumps(resultList, indent = 4))