import json
import numpy as np
import pandas as pd

with open('example_combination.json') as f:
    combinationData = json.load(f)

tmpList = [key for key in combinationData[0]]
tmpList.remove("issue")
tmpList.remove("action")
inputEvalList = tmpList

tmpList = [key for key in combinationData[0][inputEvalList[0]]]
inputModelList = tmpList

response = {}
response['combinationList'] = list(range(len(combinationData) * len(inputModelList)))
response['inputEvalList'] = inputEvalList
response['inputModelList'] = inputModelList

modelNameList = []
combinationIssueList = []
combinationActionList = []

for i in range(len(combinationData)):
    for j in range(len(inputModelList)):
        # combination - model name
        modelNameList.append(inputModelList[j])

        # combination - issue icon
        combinationIssue = list(combinationData[i]["issue"])
        combinationIssueList.append(combinationIssue)

        # combination - action icon
        combinationAction = combinationData[i]["action"]
        combinationActionList.append(combinationAction)

response['modelNames'] = modelNameList
response['combinationIconList'] = combinationIssueList
response['combinationDetailIconList'] = combinationActionList

evalList = []
for i in range(len(inputEvalList)):
    evalList.append([])

for i in range(len(combinationData)):
    for j in range(len(inputEvalList)):
        targetEval = combinationData[i][inputEvalList[j]]

        for k in range(len(inputModelList)):
            combinationEval = targetEval[inputModelList[k]]
            evalList[j].append(combinationEval)

for i in range(len(inputEvalList)):
    response[inputEvalList[i]] = evalList[i]

with open('example_combinationTable.json', 'w') as f:
    json.dump(response, f, indent = 4)