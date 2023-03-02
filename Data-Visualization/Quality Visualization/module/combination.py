import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import module.main as main
from pycaret.regression import *

uploadFileName = 'housing'
targetColumn = 'PRICE'
inputModelList = ['lr', 'rf', 'dt']
inputEvalList = ['RMSE']

# uploadFileName = 'beijing'
# targetColumn = 'PM2.5'
# inputModelList = ['lr', 'dt', 'rf', 'mlp']
# inputEvalList = ['RMSE', 'R2']

corrThreshold = 0.8
originDf = pd.read_csv(uploadFileName + '.csv')
originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
columnList = list(originDf.columns)

# completeness check
missing = sum(list(originDf.isnull().sum().values))

# outlier check based 'IQR'
tmpList = []
for column in originDf:
    df = pd.DataFrame(pd.to_numeric(originDf[column], errors = 'coerce'))
    df = df.dropna()

    lower, upper = main.lower_upper(df[column])
    lowerIdxList = list(df[df[column] > upper].index.values)
    upperIdxList = list(df[df[column] < lower].index.values)
    tmpList.append(len(lowerIdxList + upperIdxList))
    outlier = sum(tmpList)

# homogeneity check
tmpList = []
for column in originDf:
    df = originDf[column].dropna()    
    df = pd.DataFrame(pd.to_numeric(df, errors = 'coerce'))
    conIdxList = list(df[df[column].isnull()].index)
    tmpList.append(len(conIdxList))
    incons = sum(tmpList)

# duplicate check
duplicate = len(originDf[originDf.duplicated(keep = False)])

# correlation, relevance check based on 'pearson and 0.8'
tmpList = []
for column in columnList:
    df = originDf[column].dropna()
    tmpList.append(pd.to_numeric(df, errors = 'coerce'))
df = pd.concat(tmpList, axis = 1)
allCorrDf = df.corr()

# correlation
highCorrColumnList = []
for row in columnList:
    for column in columnList:
        if row == column: break
        if allCorrDf.loc[row][column] > corrThreshold or allCorrDf.loc[row][column] < -corrThreshold:
            highCorrColumnList.append([row, column])

highCorrColumnList = list(set(sum(highCorrColumnList, [])))
highCorr = len(highCorrColumnList)

# relevance
targetColumnCorrDf = abs(allCorrDf[targetColumn])

highCorrTargetColumnList = []
for row in columnList:
    if row == targetColumn: continue
    if targetColumnCorrDf[row] > corrThreshold:
        highCorrTargetColumnList.append(row)

problemColumnList = []
for highCorrColumn in highCorrTargetColumnList:
    for column in columnList:
        if allCorrDf[highCorrColumn][column] < (1 - corrThreshold) and allCorrDf[highCorrColumn][column] > -(1 - corrThreshold):
            problemColumnList.append(column)

problemCorr = len(problemColumnList)
print(missing, outlier, incons, duplicate, highCorr, problemCorr)

# to do action
actionList = []
if missing > 0:
    actionList.append('m')
if outlier > 0:
    actionList.append('o')
if incons > 0:
    actionList.append('i')
if duplicate > 0:
    actionList.append('d')
if highCorr > 0:
    actionList.append('c')
if problemCorr > 0:
    actionList.append('r')
print(actionList)

# permutation
permutationList = []
for i in range(len(actionList)):
    permutationList.append(list(map("".join, itertools.permutations(actionList, i + 1))))
permutationList = sum(permutationList, [])
print(permutationList)

# combination
missingList = ["remove", "min", "max", "mean", "median"]
outlierList = ["iqr", "zscore"]
corrList = ["pearson", "spearman", "kendall"]

totalIssueList = []
totalActionList = []
totalActionDfList = []

for permutation in permutationList:
    print(permutation)

    # for issue
    multipleCnt = 1
    if 'm' in permutation:
        multipleCnt = multipleCnt * len(missingList)
    if 'o' in permutation:
        multipleCnt = multipleCnt * len(outlierList)
    if 'c' in permutation:
        multipleCnt = multipleCnt * len(corrList)
    if 'r' in permutation:
        multipleCnt = multipleCnt * len(corrList)

    for i in range(0, multipleCnt):
        totalIssueList.append(permutation)

    # for action
    beforeActionList = ["init"]

    # for dataframe
    beforeActionDfList = [originDf]

    alphabetList = []
    for i in range(len(permutation)):
        alphabetList.append(permutation[i: i + 1])

    cnt = 0
    for alphabet in alphabetList:
        cnt = cnt + 1
        
        if alphabet == 'm':
            actionList = []
            actionDfList = []

            for j in range(len(beforeActionDfList)):
                beforeAction = beforeActionList[j]
                df = beforeActionDfList[j]
                columnList = list(df.columns)

                for action in missingList:
                    if beforeAction == "init":
                        actionList.append([action])
                    else:
                        tmpList = []
                        for i in range(len(beforeAction)):
                            tmpList.append(beforeAction[i])
                        tmpList.append(action)
                        actionList.append(tmpList)
                    
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
                                
                                ##### have to develop em, locf, knn, multiple

                            else:
                                columnDf = columnDf

                            actionColumnDfList.append(columnDf)

                        columnConcatDf = actionColumnDfList[0]
                        for k in range(len(actionColumnDfList) - 1):
                            columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[k + 1]], axis = 1, join = 'inner')
                            columnConcatDf = columnConcatDf.reset_index(drop = True)

                    columnConcatDf = columnConcatDf.reset_index(drop = True)
                    actionDfList.append(columnConcatDf)

        if alphabet == 'o':
            actionList = []
            actionDfList = []

            for j in range(len(beforeActionDfList)):
                beforeAction = beforeActionList[j]
                df = beforeActionDfList[j]
                columnList = list(df.columns)

                for action in outlierList:
                    if beforeAction == "init":
                        actionList.append([action])
                    else:
                        tmpList = []
                        for i in range(len(beforeAction)):
                            tmpList.append(beforeAction[i])
                        tmpList.append(action)
                        actionList.append(tmpList)
                    
                    actionColumnDfList = []
    
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
                    for k in range(len(actionColumnDfList) - 1):
                        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[k + 1]], axis = 1, join = 'inner')
                        columnConcatDf = columnConcatDf.reset_index(drop = True)

                    columnConcatDf = columnConcatDf.reset_index(drop = True)
                    actionDfList.append(columnConcatDf)

        if alphabet == 'i':
            actionList = []
            actionDfList = []

            for j in range(len(beforeActionDfList)):
                beforeAction = beforeActionList[j]
                df = beforeActionDfList[j]
                columnList = list(df.columns)

                if beforeAction == "init":
                    actionList.append(["remove"])
                else:
                    tmpList = []
                    for i in range(len(beforeAction)):
                        tmpList.append(beforeAction[i])
                    tmpList.append("remove")
                    actionList.append(tmpList)

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
                for k in range(len(actionColumnDfList) - 1):
                    columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[k + 1]], axis = 1, join = 'inner')
                    columnConcatDf = columnConcatDf.reset_index(drop = True)

                columnConcatDf = columnConcatDf.reset_index(drop = True)
                actionDfList.append(columnConcatDf)

        if alphabet == 'd':
            actionList = []
            actionDfList = []

            for j in range(len(beforeActionDfList)):
                beforeAction = beforeActionList[j]
                if beforeAction == "init":
                    actionList.append(["remove"])
                else:
                    tmpList = []
                    for i in range(len(beforeAction)):
                        tmpList.append(beforeAction[i])
                    tmpList.append("remove")
                    actionList.append(tmpList)

                df = beforeActionDfList[j]
                columnConcatDf = df.drop_duplicates()
                columnConcatDf = columnConcatDf.reset_index(drop = True)
                actionDfList.append(columnConcatDf)

        if alphabet == 'c' or alphabet == 'r':
            actionList = []
            actionDfList = []

            for j in range(len(beforeActionDfList)):
                beforeAction = beforeActionList[j]  
                df = beforeActionDfList[j]
                columnList = list(df.columns)

                for action in corrList:
                    if beforeAction == "init":
                        actionList.append([action])
                    else:
                        tmpList = []
                        for i in range(len(beforeAction)):
                            tmpList.append(beforeAction[i])
                        tmpList.append(action)
                        actionList.append(tmpList)

                    inconsNaNSeries = df.apply(pd.to_numeric, errors = 'coerce')
                    inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
                    allCorrDf = inconsNaNDf.corr(method = action)

                    if alphabet == 'c':
                        highCorrColumnList = []
                        for row in columnList:
                            for column in columnList:
                                if row == column: break
                                if allCorrDf.loc[row][column] > corrThreshold or allCorrDf.loc[row][column] < -corrThreshold:
                                    highCorrColumnList.append([row, column])

                        highCorrColumnList = list(set(sum(highCorrColumnList, [])))
                        if targetColumn in highCorrColumnList: highCorrColumnList.remove(targetColumn)
                        columnConcatDf = df.drop(highCorrColumnList, axis = 1)

                    if alphabet == 'r':
                        targetColumnCorrDf = abs(allCorrDf[targetColumn])

                        highCorrTargetColumnList = []
                        for row in columnList:
                            if row == targetColumn: continue
                            if targetColumnCorrDf[row] > corrThreshold:
                                highCorrTargetColumnList.append(row)

                        problemColumnList = []
                        for highCorrColumn in highCorrTargetColumnList:
                            for column in columnList:
                                if allCorrDf[highCorrColumn][column] < (1 - corrThreshold) and allCorrDf[highCorrColumn][column] > -(1 - corrThreshold):
                                    problemColumnList.append(column)
                        
                        if targetColumn in problemColumnList: problemColumnList.remove(targetColumn)
                        columnConcatDf = df.drop(problemColumnList, axis = 1)

                    actionDfList.append(columnConcatDf)
        
        beforeActionList = actionList
        beforeActionDfList = actionDfList

    for i in range(len(beforeActionList)):
        totalActionList.append(beforeActionList[i])

    for i in range(len(beforeActionDfList)):
        totalActionDfList.append(beforeActionDfList[i])

# incons, missing drop for autoML
dfList = []
for i in range(len(totalActionDfList)):
    columnList = list(totalActionDfList[i].columns)

    df = totalActionDfList[i].apply(pd.to_numeric, errors = 'coerce')
    df = pd.DataFrame(df, columns = columnList)
    df = df.dropna()
    df = df.reset_index(drop = True)

    dfList.append(df)

totalEvalList = ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE', 'TT (Sec)']
resultList = []

# kstest
originDf = pd.read_csv(uploadFileName + '.csv')
originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
columnList = list(originDf.columns)

for i in range(0, len(dfList)):
    beforeDf = originDf.apply(pd.to_numeric, errors = 'coerce')
    beforeDf = pd.DataFrame(df, columns = columnList)

    beforeDf = beforeDf[targetColumn].dropna()
    afterDf = dfList[i][targetColumn].dropna()
    kstest = abs(stats.ks_2samp(beforeDf, afterDf).pvalue)

    # distort
    if kstest > 0.05: continue

    # autoML
    clf = setup(data = dfList[i], target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
    model = compare_models(include = inputModelList)
    resultDf = pull()

    # autoML result dataframe to dict
    if len(resultDf) < len(inputModelList): continue

    for evalMetric in totalEvalList:
        if evalMetric not in inputEvalList:
            resultDf = resultDf.drop([evalMetric], axis = 1)

    resultDict = resultDf.to_dict()
    resultDict['issue'] = totalIssueList[i]
    resultDict['action'] = totalActionList[i]
    resultList.append(resultDict)

# autoML result to file
with open('result.json', 'w') as file:
    file.write(json.dumps(resultList, indent = 4))
