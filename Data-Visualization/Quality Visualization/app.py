from flask import *
from flask_cors import CORS

import os
import ast
import csv
import math
import json
import codecs
import itertools
import numpy as np
import pandas as pd
from io import StringIO
from scipy import stats
from collections import Counter
# from pycaret.regression import *

import module.main as main
import module.tree as tree

app = Flask(__name__)
CORS(app)

uploadFileName = 'housing'
targetColumn = 'PRICE'
inputModelList = ['rf', 'dt', 'lr']
inputEvalList = ['RMSE']

regModelList = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par',
                'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada',
                'gbr', 'mlp', 'lightgbm']
regEvalList = ['MAE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
combination = []
combinationDetail = []

@app.route('/fileUpload', methods=['GET', 'POST'])
def fileUpload():
  req = request.files['file']
  return json.dumps({'fileUpload': 'success'})

@app.route('/setting', methods=['GET', 'POST'])
def setting():
  global uploadFileName, regModelList, targetColumn # inputModelList, inputEvalList

  originDf = pd.read_csv('static/' + uploadFileName + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)

  if request.method == 'GET':
    columnList = []
    tmpList = list(originDf.columns)
    for i in range(len(tmpList)):
      columnList.append({'label': tmpList[i], 'value': i})

    modelList = []
    for i in range(len(regModelList)):
      modelList.append({'label': regModelList[i], 'value': i})

    evalList = []
    for i in range(len(regEvalList)):
      evalList.append({'label': regEvalList[i], 'value': i})

    response = {}
    response['columnList'] = columnList
    response['modelList'] = modelList
    response['evalList'] = evalList
    
    return json.dumps(response)

  if request.method == 'POST':
    req = eval(request.get_data().decode('utf-8'))
    targetColumn = req["column"]["label"]
    modelList = req["model"]
    evalList = req["eval"]

    inputModelList = []
    for i in range(len(modelList)):
      inputModelList.append(modelList[i]["label"])

    inputEvalList = []
    for i in range(len(evalList)):
      inputEvalList.append(evalList[i]["label"])

    return json.dumps({'setting': 'success'})

@app.route('/donutChart', methods=['GET', 'POST'])
def donutChart():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  columnList = list(originDf.columns)
  totalNum = len(originDf) * len(list(originDf.columns))

  # completeness
  mis = sum(list(originDf.isnull().sum().values))
  misRate = 100 - round((mis/totalNum) * 100)
  
  # outlier
  tmpList = []
  for column in columnList:
    df = pd.DataFrame(pd.to_numeric(originDf[column], errors = 'coerce'))
    df = df.dropna()

    lower, upper = main.lower_upper(df[column])
    lowerIdxList = list(df[df[column] > upper].index.values)
    upperIdxList = list(df[df[column] < lower].index.values)
    tmpList.append(len(lowerIdxList + upperIdxList))
  out = sum(tmpList)
  outRate = 100 - round((out/totalNum) * 100)

  # homogeneity
  tmpList = []
  for column in columnList:
    df = originDf[column].dropna()
    df = pd.DataFrame(pd.to_numeric(df, errors = 'coerce'))
    conIdxList = list(df[df[column].isnull()].index)
    tmpList.append(len(conIdxList))
  inc = sum(tmpList)
  incRate = 100 - round((inc/totalNum) * 100)

  # duplicate
  dupCnt = len(originDf[originDf.duplicated(keep = False)])
  dupRate = 100 - round(dupCnt/len(originDf) * 100)

  inconsNaNSeries = originDf.apply(pd.to_numeric, errors = 'coerce')
  inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
  allCorrDf = inconsNaNDf.corr()
  allCorrDf = allCorrDf.fillna(0)

  highCorr = 0
  corrThreshold = 0.8

  # correlation
  highCorrColumnList = []
  for column in columnList:
    columnCorrDf = abs(allCorrDf[column])
    highCorrDf = columnCorrDf[columnCorrDf > corrThreshold]
    
    if len(highCorrDf) > 1:
      highCorrColumnList.append(list(highCorrDf.index))

  highCorrColumnList = list(set([tuple(set(item)) for item in highCorrColumnList]))
  highCorr = len(highCorrColumnList) * 2
  corRate = 100 - round(highCorr/len(columnList) * 100)

  global targetColumn

  # relevance
  columnCorrDf = abs(allCorrDf[targetColumn])

  highCorrColumnList = []
  for row in columnList:
    if row == targetColumn: continue
    if columnCorrDf[row] < corrThreshold:
      highCorrColumnList.append(row)
  
  highColumnCorr = len(highCorrColumnList)
  relRate = 100 - round(highColumnCorr/(len(columnList) - 1) * 100)

  rateList = [misRate, outRate, incRate, dupRate, corRate, relRate]
  colorList = ['darkorange', 'steelblue', 'yellowgreen', 'lightcoral', 'darkslategray', 'mediumpurple']

  donutChartList = []
  for i in range(0, 6):
    donutChartList.append({'label': i, 'color': colorList[i], 'data': {'issue': rateList[i], 'normal': 100 - rateList[i]}})

  response = {}
  response['donutChartData'] = donutChartList

  return json.dumps(response)

@app.route('/checkVisualization', methods=['GET', 'POST'])
def checkVisualization():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]
  vis = req["visualization"]
  metric = req["metricValue"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  columnList = list(originDf.columns)

  global targetColumn
  response = {'visualization': vis}
  
  # completeness, homogeneity
  if vis == 'heatmapChart':
    sliceCnt = 10
    sliceSize = int(len(originDf)/sliceCnt)

    seriesDataList = []
    for i in range(sliceCnt):
      rowSliceDf = originDf.iloc[sliceSize * i : sliceSize * (i + 1)]
      columnCntList = []

      for column in columnList:
        columnDf = rowSliceDf[column]

        if metric == 'homogeneity':
          columnDf = rowSliceDf[column]
          columnDf = columnDf.dropna()
          columnDf = pd.to_numeric(columnDf, errors = 'coerce')

        columnCnt = columnDf.isnull().sum()
        columnCntList.append(int(columnCnt))
      seriesDataList.append({'name': 'r' + str(i), 'data': columnCntList})
      categoryDataList = []
      for i in range(len(columnList)):
        categoryDataList.append('f' + str(i))

    response['seriesData'] = seriesDataList
    response['categoryData'] = categoryDataList

    rowIdx = req["rowIdx"]
    columnIdx = req["columnIdx"]

    response['rowIndex'] = str(sliceSize * int(rowIdx)) + '~' + str(sliceSize * (int(rowIdx) + 1))
    response['columnName'] = columnList[int(columnIdx)]

    columnDf = originDf.iloc[:, [columnIdx]]
    sliceDf = columnDf.iloc[sliceSize * int(rowIdx) : sliceSize * (int(rowIdx) + 1)]
    missingIdx = [index for index, row in sliceDf.iterrows() if row.isnull().any()]

    if metric == 'completeness': issue = 'NaN'
    if metric == 'homogeneity': issue = 'incons'

    issueList = []
    for i in missingIdx:
      issueList.append([i, issue])

    response['issueList'] = issueList

  # outlier
  if vis == 'histogramChart':
    method = req["outlier"]
    columnName = req["column"]

    columnDf = originDf[columnName]
    columnDf = columnDf.apply(pd.to_numeric, errors = 'coerce')

    columnList = columnDf.values.tolist()
    minValue = columnDf.min()
    maxValue = columnDf.max()

    sliceCnt = 20
    sliceSize = (maxValue - minValue)/sliceCnt
    columnCntList = [0 for i in range(sliceCnt)]

    seriesDataList = []
    categoryDataList = []

    for i in range(sliceCnt):
      minRange = float(minValue + (sliceSize * i))
      maxRange = float(minValue + (sliceSize * (i + 1)))
      categoryDataList.append(maxRange)

      for j in range(len(columnList)):
        if math.isnan(columnList[j]) == False:
          if columnList[j] >= minRange and columnList[j] <= maxRange:
            columnCntList[i] = columnCntList[i] + 1

    seriesDataList.append({'name': columnName, 'data': columnCntList})

    response['seriesData'] = seriesDataList
    response['categoryData'] = categoryDataList

    if method == 'iqr':
      lower, upper = main.lower_upper(columnDf)
      lowerIdxList = list(columnDf[columnDf > upper].index.values)
      upperIdxList = list(columnDf[columnDf < lower].index.values)
      outlierIndex = lowerIdxList + upperIdxList

      response['cnt'] = len(outlierIndex)
      response['lower'] = round(lower, 3)
      response['upper'] = round(upper, 3)
      response['standard'] = 'lower than ' + str(round(lower, 3)) + ', higher than ' + str(round(upper, 3))

      issueList = []
      for i in outlierIndex:
        outlier = columnDf.iloc[i]
        issueList.append([str(i), str(round(outlier, 3))])
      
      response['issueList'] = issueList

    if method == 'z-score':
      columnSeries = columnDf.dropna()
      meanValue = np.mean(columnSeries)
      stdValue = np.std(columnSeries)

      outlierIndex = []
      zscoreThreshold = 3
      for i in range(len(columnDf)):
        value = columnDf.iloc[i]
        zscore = ((value - meanValue)/stdValue)

        if zscore > zscoreThreshold:
          outlierIndex.append(i)
      response['cnt'] = len(outlierIndex)

      issueList = []
      outlierValue = []
      for i in outlierIndex:
        outlier = columnDf.iloc[i]
        outlierValue.append(outlier)
        issueList.append([str(i), str(round(outlier, 3))])
      
      response['threshold'] = round(min(outlierValue), 3)
      response['standard'] = 'greater than ' + str(round(min(outlierValue), 3))
      response['issueList'] = issueList

  # duplicate
  if vis == 'duplicate':
    dupDf = originDf[originDf.duplicated(keep = False)]
    dupList = list(dupDf.index)
    dupList = list(map(str, dupList))
    issueList = ', '.join(dupList)

    response['issueList'] = issueList
    response['cnt'] = len(dupList)

  if vis == 'correlationChart' or vis == 'relevanceChart':
    method = req["method"]
    corrThreshold = 0.8

    inconsNaNSeries = originDf.apply(pd.to_numeric, errors = 'coerce')
    inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
    allCorrDf = inconsNaNDf.corr(method = method)
    allCorrDf = allCorrDf.fillna(0)
    allCorrDf = allCorrDf.reindex(sorted(allCorrDf.columns), axis = 1)

    # correlation
    if vis == 'correlationChart':
      highCorrColumnList = []
      for column in columnList:
        columnCorrDf = abs(allCorrDf[column])
        highCorrDf = columnCorrDf[columnCorrDf > corrThreshold]
        
        if len(highCorrDf) > 1:
          highCorrColumnList.append(list(highCorrDf.index))
      highCorrColumnList = list(set([tuple(set(item)) for item in highCorrColumnList]))

      response['cnt'] = len(highCorrColumnList) * 2
      response['issueList'] = highCorrColumnList

      seriesDataList = []
      for i in range(len(allCorrDf)):
        columnCntList = []

        for j in range(i + 1):
          columnCntList.append(float(allCorrDf.iloc[i][j]))
        seriesDataList.append({'name': 'f' + str(i), 'data': columnCntList})

      categoryDataList = []
      for i in range(len(columnList)):
        categoryDataList.append('f' + str(i))

      response['seriesData'] = seriesDataList
      response['categoryData'] = categoryDataList

    # relevance
    if vis == 'relevanceChart':
      columnCorrDf = allCorrDf[targetColumn]

      seriesDataList = []
      highCorrColumnList = []
      for row in columnList:
        if row == targetColumn: continue
        if columnCorrDf[row] < corrThreshold and columnCorrDf[row] > -corrThreshold:
          highCorrColumnList.append(row)
        seriesDataList.append(columnCorrDf[row])

      response['cnt'] = len(highCorrColumnList)
      response['issueList'] = highCorrColumnList
      response['seriesData'] = seriesDataList
      response['categoryData'] = columnList

  return json.dumps(response)

@app.route('/modelTable', methods=['GET', 'POST'])
def modelTable():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  # originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  # originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  # columnList = list(originDf.columns)

  # df = originDf.apply(pd.to_numeric, errors = 'coerce')
  # df = pd.DataFrame(df, columns = columnList)
  # df = df.dropna()

  # global targetColumn, regModelList
  # clf = setup(data = df, target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
  # model = compare_models(include = regModelList)
  # modelResultDf = pull()

  # modelResultDf = modelResultDf.drop(['Model', 'MSE', 'TT (Sec)'], axis = 1)
  # modelResultDf['Model'] = modelResultDf.index
  
  # firstColumnList = list(modelResultDf.columns[-1:])
  # remainColumnList = list(modelResultDf.columns[:-1])
  # arrangeColumnList = firstColumnList + remainColumnList

  # modelResultDf = modelResultDf[arrangeColumnList]
  # modelResultDf = modelResultDf.round(3)

  # modelResultDf.to_csv('static/example_modelTable.csv', index = False)
  modelResultDf = pd.read_csv('static/example_modelTable_step2.csv')

  modelResultList = [list(modelResultDf.columns)]
  for i in range(len(modelResultDf)):
    modelResultList.append(list(modelResultDf.iloc[i]))

  response = {}
  response['modelResultData'] = modelResultList

  return json.dumps(response)

@app.route('/tableData', methods=['GET', 'POST'])
def tableData():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  originDf = originDf.fillna('')
  columnList = list(originDf.columns)

  originDfList = originDf.values.tolist()
  originDfList.insert(0, columnList)

  response = {}
  response['tableData'] = originDfList

  return json.dumps(response)

@app.route('/tablePoint', methods=['GET', 'POST'])
def tablePoint():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  columnList = list(originDf.columns)
  
  misPointList = []
  for column in columnList:
    misIdxList = list(originDf[originDf[column].isnull()].index)

    if len(misIdxList) > 0:
      for row in misIdxList:
        misPointList.append({'x': str(row + 1), 'y': columnList.index(column)})

  outPointList = []
  for column in columnList:
    df = pd.DataFrame(pd.to_numeric(originDf[column], errors = 'coerce'))
    df = df.dropna()

    lower, upper = main.lower_upper(df[column])
    lowerIdxList = list(df[df[column] > upper].index.values)
    upperIdxList = list(df[df[column] < lower].index.values)
    outIdxList = lowerIdxList + upperIdxList

    if len(outIdxList) > 0:
      for row in outIdxList:
        outPointList.append({'x': str(row + 1), 'y': columnList.index(column)})

  conPointList = []
  for column in columnList:  
    df = originDf[column].dropna()
    df = pd.DataFrame(pd.to_numeric(df, errors = 'coerce'))
    conIdxList = list(df[df[column].isnull()].index)

    if len(conIdxList) > 0:
      for row in conIdxList:
        conPointList.append({'x': str(row + 1), 'y': columnList.index(column)})

  response = {}
  response['comPointList'] = misPointList
  response['accPointList'] = outPointList
  response['conPointList'] = conPointList

  return json.dumps(response)

@app.route('/columnSummary', methods=['GET', 'POST'])
def columnSummary():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  columnList = list(originDf.columns)
  totalNum = len(originDf)

  inconsNaNSeries = originDf.apply(pd.to_numeric, errors = 'coerce')
  inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
  allCorrDf = inconsNaNDf.corr(method = 'kendall')
  allCorrDf = allCorrDf.fillna(0)
  
  corrThreshold = 0.8

  # correlation
  corrColumnList = []
  for column in columnList:
    columnCorrDf = abs(allCorrDf[column])
    highCorrDf = columnCorrDf[columnCorrDf > corrThreshold]
    
    if len(highCorrDf) > 1:
      corrColumnList.append(list(highCorrDf.index))

  corrColumnList = sum(corrColumnList, [])
  corrColumnList = list(set(corrColumnList))
  corList = [0 for i in range(len(columnList))]
  for i in range(len(columnList)):
    if columnList[i] in corrColumnList:
      corList[i] = 100

  global targetColumn

  # relevence
  allCorrDf = allCorrDf.reindex(sorted(allCorrDf.columns), axis = 1)
  columnCorrDf = allCorrDf[targetColumn]

  columnCorrColumnList = []
  for row in columnList:
    if row == targetColumn: continue
    if columnCorrDf[row] < corrThreshold and columnCorrDf[row] > -corrThreshold:
      columnCorrColumnList.append(row)

  relList = [0 for i in range(len(columnList))]
  for i in range(len(columnList)):
    if columnList[i] in columnCorrColumnList:
      relList[i] = 100

  seriesData = []
  columnSummaryData = [relList, corList]
  for i in range(0, 2):
    seriesData.append({'name': 'r' + str(i + 1), 'data': columnSummaryData[i]})
  
  response = {}
  response['seriesData'] = seriesData
  response['categoryData'] = columnList

  return json.dumps(response)

@app.route('/rowSummary', methods=['GET', 'POST'])
def rowSummary():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  dupDf = originDf[originDf.duplicated(keep = False)]
  dupList = list(dupDf.index)

  response = {}
  response['rowIndex'] = dupList

  return json.dumps(response)

@app.route('/combination', methods=['GET', 'POST'])
def combinationTable():
  with open('static/example_combinationTable.json') as f:
    combinationData = json.load(f)

  return json.dumps(combinationData)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
  req = request.get_data().decode('utf-8')
  req = eval(req)

  global uploadFileName, combination, combinationDetail, targetColumn
  combination = req["combination"]
  combinationDetail = req["combinationDetail"]

  originDf = pd.read_csv('static/' + uploadFileName + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)

  for file in os.scandir('static/dataset/'):
    os.remove(file.path)

  beforeDf = originDf
  beforeDf.to_csv('static/dataset/0.csv', index = False)

  for i in range(len(combination)):
    columnList = list(beforeDf.columns)

    issue = combination[i]
    action = combinationDetail[i]

    if issue == 'completeness':
      actionColumnDfList = []

      if action == "remove":
        columnConcatDf = beforeDf.dropna()
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      else:
        for column in columnList:
          columnDf = beforeDf.loc[:, [column]]
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
        for j in range(len(actionColumnDfList) - 1):
          columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
          columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'outlier':
      actionColumnDfList = []

      for column in columnList:
        columnDf = beforeDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        if action == 'iqr':
          lower, upper = main.lower_upper(inconsNaNDf)
          lowerIdxList = list(inconsNaNDf[inconsNaNDf[column] > upper].index.values)
          upperIdxList = list(inconsNaNDf[inconsNaNDf[column] < lower].index.values)
          outlierIndex = lowerIdxList + upperIdxList

        if action == 'zscore':
          inconsNaNSeries = inconsNaNDf.dropna()
          meanValue = np.mean(inconsNaNSeries)
          stdValue = np.std(inconsNaNSeries)

          outlierIndex = []
          zscoreThreshold = 3
          for j in range(len(inconsNaNDf)):
              value = inconsNaNDf.iloc[j].values[0]
              zscore = ((value - meanValue)/stdValue).values[0]

              if zscore > zscoreThreshold:
                outlierIndex.append(j)
      
        for idx in missingIndex:
          columnDf.loc[idx, column] = np.nan

        for idx in inconsIndex:
          columnDf.loc[idx, column] = 'incons'
        
        columnDf = columnDf.drop(outlierIndex)
        actionColumnDfList.append(columnDf)

      columnConcatDf = actionColumnDfList[0]
      for j in range(len(actionColumnDfList) - 1):
        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'homogeneity':
      actionColumnDfList = []

      for column in columnList:
        columnDf = beforeDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]
        
        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        columnDf = columnDf.drop(inconsIndex)
        actionColumnDfList.append(columnDf)

      columnConcatDf = actionColumnDfList[0]
      for j in range(len(actionColumnDfList) - 1):
        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'duplicate':
      df = beforeDf.drop_duplicates()
      df = df.reset_index(drop = True)

      beforeDf = df

    if issue == 'correlation' or issue == 'relevance':
      inconsNaNSeries = beforeDf.apply(pd.to_numeric, errors = 'coerce')
      inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
      allCorrDf = inconsNaNDf.corr(method = action)
      allCorrDf = allCorrDf.fillna(0)
      corrThreshold = 0.8

      highCorrList = []
      if issue == 'correlation':
        for row in columnList:
          for column in columnList:
            if row == column: break
            if allCorrDf.loc[row][column] > corrThreshold or allCorrDf.loc[row][column] < -corrThreshold:
              highCorrList.append([row, column])

        highCorrList = list(set(sum(highCorrList, [])))

      if issue == 'relevance':
        columnCorrDf = allCorrDf[targetColumn]
        
        for row in columnList:
          if columnCorrDf[row] < corrThreshold and columnCorrDf[row] > -corrThreshold:
            highCorrList.append(row)

      if targetColumn in highCorrList: highCorrList.remove(targetColumn)
      beforeDf = beforeDf.drop(highCorrList, axis = 1)

    beforeDf.to_csv('static/dataset/' + str(i + 1) + '.csv', index = False)

  return json.dumps({'recommend': 'success'})

@app.route('/newVisualization', methods=['GET', 'POST'])
def newVisualization():
  req = request.get_data().decode('utf-8')
  req = eval(req)
  
  fileName = req["fileName"]
  select = req["select"]
  selectDetail = req["selectDetail"]

  originDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)

  # visualization
  if select == 'column':
    columnDf = originDf.loc[:, [selectDetail]]
    columnDf = columnDf.apply(pd.to_numeric, errors = 'coerce')

    # box plot
    columnSeries = columnDf.dropna()
    q1 = np.quantile(columnSeries, 0.25)
    q3 = np.quantile(columnSeries, 0.75)
    iqr = q3 - q1
    q2 = np.quantile(columnSeries, 0.5)
    q4 = np.quantile(columnSeries, 1)    

    data = {}
    data['x'] = selectDetail
    data['y'] = [q1, q2, iqr, q3, q4]

    response = {}
    response['boxplotSeriesData'] = [data]

    # histogram
    columnDf = originDf[selectDetail]
    columnDf = columnDf.apply(pd.to_numeric, errors = 'coerce')

    columnList = columnDf.values.tolist()
    minValue = columnDf.min()
    maxValue = columnDf.max()

    sliceCnt = 20
    sliceSize = (maxValue - minValue)/sliceCnt
    columnCntList = [0 for i in range(sliceCnt)]

    seriesDataList = []
    categoryDataList = []

    for i in range(sliceCnt):
      minRange = float(minValue + (sliceSize * i))
      maxRange = float(minValue + (sliceSize * (i + 1)))
      categoryDataList.append(maxRange)

      for j in range(len(columnList)):
        if math.isnan(columnList[j]) == False:
          if columnList[j] >= minRange and columnList[j] <= maxRange:
            columnCntList[i] = columnCntList[i] + 1

    seriesDataList.append({'name': selectDetail, 'data': columnCntList})

    response['histogramSeriesData'] = seriesDataList
    response['histogramCategoryData'] = categoryDataList

  if select == 'row':
    # scatter plot
    scatterDf = originDf.apply(pd.to_numeric, errors = 'coerce').dropna()
    scatterDf = scatterDf.reset_index(drop = True)

    from sklearn.manifold import TSNE
    dataMatrix = scatterDf.values

    tsneDf = TSNE(n_components = 2, random_state = 0).fit_transform(dataMatrix)
    tsneDf = pd.DataFrame(tsneDf, columns = ['value1', 'value2'])
    tsneList = tsneDf.values.tolist()

    selectList = [tsneList[int(selectDetail)]]
    del tsneList[int(selectDetail)]

    response = {}
    response['selectSeriesData'] = selectList
    response['notSelectSeriesData'] = tsneList

  return json.dumps(response)

@app.route('/new', methods=['GET', 'POST'])
def new():
  req = request.get_data().decode('utf-8')
  req = eval(req)
  
  fileName = req["fileName"]
  select = req["select"]
  selectDetail = req["selectDetail"]
  action = req["action"]

  originDf = pd.read_csv('static/dataset/' + str(fileName - 1) + '.csv')
  originDf = originDf.reindex(sorted(originDf.columns), axis = 1)
  columnList = list(originDf.columns)

  global combination, combinationDetail, targetColumn
  customIssue = combination[fileName - 1]
  originAction = combinationDetail[fileName - 1]

  # customization
  if select == 'column':
    if customIssue == 'correlation' or customIssue == 'relevance':
      originDf = originDf.drop([selectDetail], axis = 1)

    if customIssue == 'homogeneity':
      actionColumnDfList = []

      for column in columnList:
        columnDf = beforeDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]
        
        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        if column == selectDetail:
          columnDf = columnDf
        else:
          columnDf = columnDf.drop(inconsIndex)
        actionColumnDfList.append(columnDf)

      columnConcatDf = actionColumnDfList[0]
      for i in range(len(actionColumnDfList) - 1):
        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[i + 1]], axis = 1, join = 'inner')
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      originDf = columnConcatDf

    if customIssue == 'completeness':
      actionColumnDfList = []

      if action == 'remove':
        originDf = originDf[selectDetail].dropna()
        originDf = originDf.reset_index(drop = True)

      else:
        for column in columnList:
          columnDf = originDf.loc[:, [column]]
          missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

          if len(missingIndex) > 0:
            inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
            inconsDropDf = inconsNaNDf.dropna()

            if column == selectDetail:
              if action == 'none':
                columnDf = columnDf
              if action == 'min':
                columnDf = main.imp_min(columnDf, inconsDropDf)
              if action == 'max':
                columnDf = main.imp_max(columnDf, inconsDropDf)
              if action == 'mean':
                columnDf = main.imp_mean(columnDf, inconsDropDf)
              if action == 'median':
                columnDf = main.imp_median(columnDf, inconsDropDf)

            else:
              if originAction == 'min':
                columnDf = main.imp_min(columnDf, inconsDropDf)
              if originAction == 'max':
                columnDf = main.imp_max(columnDf, inconsDropDf)
              if originAction == 'mean':
                columnDf = main.imp_mean(columnDf, inconsDropDf)
              if originAction == 'median':
                columnDf = main.imp_median(columnDf, inconsDropDf)
          
          else:
            columnDf = columnDf
          
          actionColumnDfList.append(columnDf)

        columnConcatDf = actionColumnDfList[0]
        for i in range(len(actionColumnDfList) - 1):
          columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[i + 1]], axis = 1, join = 'inner')
          columnConcatDf = columnConcatDf.reset_index(drop = True)

      originDf = columnConcatDf
    
    if customIssue == 'outlier':
      actionColumnDfList = []

      for column in columnList:
        columnDf = originDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        if column == selectDetail:
          if action == 'none':
            columnDf = columnDf

          if action == 'iqr':
            lower, upper = main.lower_upper(inconsNaNDf)
            lowerIdxList = list(inconsNaNDf[inconsNaNDf[column] > upper].index.values)
            upperIdxList = list(inconsNaNDf[inconsNaNDf[column] < lower].index.values)
            outlierIndex = lowerIdxList + upperIdxList

          if action == 'zscore':
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

        else:
          if originAction == 'iqr':
            lower, upper = main.lower_upper(inconsNaNDf)
            lowerIdxList = list(inconsNaNDf[inconsNaNDf[column] > upper].index.values)
            upperIdxList = list(inconsNaNDf[inconsNaNDf[column] < lower].index.values)
            outlierIndex = lowerIdxList + upperIdxList

          if originAction == 'zscore':
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

      originDf = columnConcatDf

  if select == 'row':
    originDf = originDf.drop([int(selectDetail)])

  originDf.to_csv('static/dataset/' + str(fileName) + '.csv', index = False)

  # customization after dataset
  beforeDf = originDf

  for i in range(fileName, len(combination)):
    issue = combination[i]
    action = combinationDetail[i]

    if issue == 'completeness':
      actionColumnDfList = []

      if action == "remove":
        columnConcatDf = beforeDf.dropna()
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      else:
        for column in columnList:
          columnDf = beforeDf.loc[:, [column]]
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
        for j in range(len(actionColumnDfList) - 1):
          columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
          columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'outlier':
      actionColumnDfList = []

      for column in columnList:
        columnDf = beforeDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]

        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        if action == 'iqr':
          lower, upper = main.lower_upper(inconsNaNDf)
          lowerIdxList = list(inconsNaNDf[inconsNaNDf[column] > upper].index.values)
          upperIdxList = list(inconsNaNDf[inconsNaNDf[column] < lower].index.values)
          outlierIndex = lowerIdxList + upperIdxList

        if action == 'zscore':
          inconsNaNSeries = inconsNaNDf.dropna()
          meanValue = np.mean(inconsNaNSeries)
          stdValue = np.std(inconsNaNSeries)

          outlierIndex = []
          zscoreThreshold = 3
          for j in range(len(inconsNaNDf)):
              value = inconsNaNDf.iloc[j].values[0]
              zscore = ((value - meanValue)/stdValue).values[0]

              if zscore > zscoreThreshold:
                outlierIndex.append(j)
      
        for idx in missingIndex:
          columnDf.loc[idx, column] = np.nan

        for idx in inconsIndex:
          columnDf.loc[idx, column] = 'incons'
        
        columnDf = columnDf.drop(outlierIndex)
        actionColumnDfList.append(columnDf)

      columnConcatDf = actionColumnDfList[0]
      for j in range(len(actionColumnDfList) - 1):
        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'homogeneity':
      actionColumnDfList = []

      for column in columnList:
        columnDf = beforeDf.loc[:, [column]]
        missingIndex = [index for index, row in columnDf.iterrows() if row.isnull().any()]
        
        inconsNaNDf = pd.DataFrame(pd.to_numeric(columnDf.squeeze(), errors = 'coerce'))
        missingAndInconsIndex = [index for index, row in inconsNaNDf.iterrows() if row.isnull().any()]
        inconsIndex = list(set(missingAndInconsIndex) - set(missingIndex))

        columnDf = columnDf.drop(inconsIndex)
        actionColumnDfList.append(columnDf)

      columnConcatDf = actionColumnDfList[0]
      for j in range(len(actionColumnDfList) - 1):
        columnConcatDf = pd.concat([columnConcatDf, actionColumnDfList[j + 1]], axis = 1, join = 'inner')
        columnConcatDf = columnConcatDf.reset_index(drop = True)

      beforeDf = columnConcatDf

    if issue == 'duplicate':
      df = beforeDf.drop_duplicates()
      df = df.reset_index(drop = True)

      beforeDf = df

    if issue == 'correlation' or issue == 'relevance':
      inconsNaNSeries = beforeDf.apply(pd.to_numeric, errors = 'coerce')
      inconsNaNDf = pd.DataFrame(inconsNaNSeries, columns = columnList)
      allCorrDf = inconsNaNDf.corr(method = action)
      allCorrDf = allCorrDf.fillna(0)
      corrThreshold = 0.8

      highCorrList = []
      if issue == 'correlation':
        for row in columnList:
          for column in columnList:
            if row == column: break
            if allCorrDf.loc[row][column] > corrThreshold or allCorrDf.loc[row][column] < -corrThreshold:
              highCorrList.append([row, column])

        highCorrList = list(set(sum(highCorrList, [])))
        if targetColumn in highCorrList: highCorrList.remove(targetColumn)
        beforeDf = beforeDf.drop(highCorrList, axis = 1)

      if issue == 'relevance':
        columnCorrDf = allCorrDf[targetColumn]
        
        for row in columnList:
          if columnCorrDf[row] < corrThreshold and columnCorrDf[row] > -corrThreshold:
            highCorrList.append(row)

        if targetColumn in highCorrList: highCorrList.remove(targetColumn)
        beforeDf = beforeDf.drop(highCorrList, axis = 1)

    beforeDf.to_csv('static/dataset/' + str(i + 1) + '.csv', index = False)

  return json.dumps({'new': 'success'})

@app.route('/impact', methods=['GET', 'POST'])
def impact():
  with open('static/example_after.json') as f: afterData = json.load(f)
  with open('static/example_before.json') as f: beforeData = json.load(f)

  global inputModelList, inputEvalList

  # missing
  missingList = []
  for i in range(0, 5):
    after = afterData[i][inputEvalList[0]][inputModelList[0]]
    before = beforeData[0][inputEvalList[0]][inputModelList[0]]
    impact = after - before

    missingList.append(impact)
  missing = sum(missingList)/len(missingList)

  # outlier
  outlierList = []
  for i in range(5, 7):
    after = afterData[i][inputEvalList[0]][inputModelList[0]]
    before = beforeData[0][inputEvalList[0]][inputModelList[0]]
    impact = after - before

    outlierList.append(impact)
  outlier = sum(outlierList)/len(outlierList)

  # incons
  after = afterData[7][inputEvalList[0]][inputModelList[0]]
  before = beforeData[0][inputEvalList[0]][inputModelList[0]]
  incons = after - before

  # duplicate
  after = afterData[8][inputEvalList[0]][inputModelList[0]]
  before = beforeData[0][inputEvalList[0]][inputModelList[0]]
  duplicate = after - before

  # correlation
  corrList = []
  for i in range(9, 12):
    after = afterData[i][inputEvalList[0]][inputModelList[0]]
    before = beforeData[0][inputEvalList[0]][inputModelList[0]]
    impact = after - before

    corrList.append(impact)
  corr = sum(corrList)/len(corrList)

  # relevance
  relList = []
  for i in range(12, 15):
    after = afterData[i][inputEvalList[0]][inputModelList[0]]
    before = beforeData[0][inputEvalList[0]][inputModelList[0]]
    impact = after - before

    relList.append(impact)
  rel = sum(relList)/len(relList)

  response = {}
  response['seriesData'] = [missing, outlier, incons, duplicate, corr, rel]

  return json.dumps(response)

@app.route('/changeCnt', methods=['GET', 'POST'])
def changeCnt():
  global uploadFileName
  beforeDf = pd.read_csv('static/' + uploadFileName + '.csv')
  beforeList = [len(beforeDf), len(beforeDf.columns), len(beforeDf) * len(beforeDf.columns)]
  
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  afterDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  afterList = [len(afterDf), len(afterDf.columns), len(afterDf) * len(afterDf.columns)]

  diffList = []
  for i in range(0, 3):
    diffList.append(beforeList[i] - afterList[i])

  seriesDataList = []
  seriesDataList.append({'name': 'diff', 'data': diffList})

  response = {}
  response['seriesData'] = seriesDataList

  return json.dumps(response)

@app.route('/changePerformance', methods=['GET', 'POST'])
def changePerformance():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]
  modelName = req["modelName"]

  # global uploadFileName, targetColumn, regModelList
  # beforeDf = pd.read_csv('static/' + uploadFileName + '.csv')
  # columnList = list(beforeDf.columns)

  # df = beforeDf.apply(pd.to_numeric, errors = 'coerce')
  # df = pd.DataFrame(df, columns = columnList)
  # df = df.dropna()

  # clf = setup(data = df, target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
  # model = compare_models(include = [modelName])
  # modelResultDf = pull()

  # modelResultDf = modelResultDf.drop(['Model', 'MSE', 'TT (Sec)'], axis = 1)
  # modelResultDf = modelResultDf.round(3)
  # beforeList = modelResultDf.loc[[modelName], :].values.tolist()[0]

  # afterDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  # columnList = list(beforeDf.columns)

  # df = beforeDf.apply(pd.to_numeric, errors = 'coerce')
  # df = pd.DataFrame(df, columns = columnList)
  # df = df.dropna()

  # clf = setup(data = df, target = targetColumn, preprocess = False, session_id = 42, use_gpu = True, silent = True)
  # model = compare_models(include = [modelName])
  # modelResultDf = pull()

  # modelResultDf = modelResultDf.drop(['Model', 'MSE', 'TT (Sec)'], axis = 1)
  # modelResultDf = modelResultDf.round(3)
  # afterList = modelResultDf.loc[[modelName], :].values.tolist()[0]

  # house pricing dataset - step 0
  # beforeList = [2.481, 3.592, 0.824, 0.155, 0.123]
  # afterList = [2.481, 3.592, 0.824, 0.155, 0.123]

  # house pricing dataset - step 2
  beforeList = [2.481, 3.592, 0.824, 0.155, 0.123]
  afterList = [2.341, 3.124, 0.644, 0.122, 0.1]

  seriesDataList = []
  seriesDataList.append({'name': 'before', 'data': beforeList})
  seriesDataList.append({'name': 'after', 'data': afterList})

  response = {}
  response['seriesData'] = seriesDataList

  return json.dumps(response)

@app.route('/changeDistort', methods=['GET', 'POST'])
def changeDistort():
  req = eval(request.get_data().decode('utf-8'))
  fileName = req["fileName"]

  indexList = []
  xList = []
  yList = []

  global uploadFileName, targetColumn
  beforeDf = pd.read_csv('static/' + uploadFileName + '.csv')
  beforeDf = beforeDf.apply(pd.to_numeric, errors = 'coerce')
  beforeColumnDf = beforeDf[targetColumn].dropna()

  x = np.sort(beforeColumnDf)
  y = 1. * np.arange(len(x))/float(len(x) - 1)
  for i in range(0, len(x)):
    indexList.append('before')
    xList.append(list(x)[i])
    yList.append(y[i])

  afterDf = pd.read_csv('static/dataset/' + str(fileName) + '.csv')
  afterDf = afterDf.apply(pd.to_numeric, errors = 'coerce')
  afterColumnDf = afterDf[targetColumn].dropna()

  x = np.sort(afterColumnDf)
  y = 1. * np.arange(len(x))/float(len(x) - 1)
  for i in range(0, len(x)):
    indexList.append('after')
    xList.append(list(x)[i])
    yList.append(y[i])

  resultList = []
  for i in range(0, len(indexList)):
    resultList.append({'index': indexList[i], 'x': xList[i], 'y': yList[i]})

  kstest = round(abs(stats.ks_2samp(beforeColumnDf, afterColumnDf).pvalue), 3)

  response = {}
  response['ECDFchartData'] = resultList
  response['KStestValue'] = kstest

  return json.dumps(response)

if __name__ == '__main__':
  app.jinja_env.auto_reload = True
  app.config['TEMPLATES_AUTO_RELOAD'] = True
  app.run(debug = True)