import sys
import os
import csv
import math
import json
from datetime import datetime

import cv2
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans

################################
# clustering related functions #
################################
def label_groundTruthFixationMap(_gt, _x, _y):
  if np.array_equal(_gt[_y][_x], np.array([0, 0, 0])):
    return 0
  else:
    return 1

def generate_discrete_groundTruthFixationMap(_gt):
  _dgtfmPath = "./static/__cache__/discrete_ground_truth_fixation_map.png"
  gtCopy = _gt.copy()
  mapHeight, mapWidth = gtCopy.shape[:2]
  for i in range(0, len(gtCopy)):
    for j in range(0, len(gtCopy[i])):
      replaceArr = np.array([255, 255, 255])
      if np.array_equal(gtCopy[i][j], np.array([0, 0, 0])):
        replaceArr = np.array([0, 0, 0])
      gtCopy[i][j] = replaceArr
  gtCopy.imwrite(_dgtfmPath)

def getFeatureMeanVal(_featDF, _x, _y, _stiWidth, _stiHeight, _patchSize):
  meanVal = 0
  min_x = int(_x - _patchSize/2)
  max_x = int(_x + _patchSize/2)
  if min_x < 0:
    min_x = 0
  if max_x > _stiWidth-1:
    max_x = int(_stiWidth-1)
  min_y = int(_y - _patchSize/2)
  max_y = int(_y + _patchSize/2)
  if min_y < 0:
    min_y = 0
  if max_y > _stiHeight-1:
    max_y = int(_stiHeight-1)
  featNP = _featDF.to_numpy()
  # print("top: %d, bottom: %d, left: %d, right: %d"%(min_y, max_y, min_x, max_x))
  patch = featNP[min_y:max_y, min_x:max_x]
  # print(patch.shape)
  meanVal = patch.mean()
  return meanVal

def dataTransformation(tMethod, df, featureList):
  # print("Data transformation method: "+tMethod)
  if tMethod == "raw":
    return df
  elif tMethod == "min_max":
    return dt_minMax(df, featureList)
  elif tMethod == "z_score":
    return dt_zScore(df, featureList)
  elif tMethod == "yeo_johonson":
    return dt_yeoJohnson(df, featureList)
  elif tMethod == "yeo_johonson_min_max":
    return dt_yeoJohnson_minMax(df, featureList)
  else:
    print("ERROR: unavailable data transformation method selected")
    return df

def dt_minMax(df, featureList):
  getColNames = df.columns
  tfDF = df[[getColNames[0], getColNames[1], getColNames[2], getColNames[3]]]
  colCount = 3
  for featureName in featureList:
    colCount = colCount+1
    colFeatDF = df[featureName]
    scaler = MinMaxScaler()
    _tf = scaler.fit_transform(colFeatDF.values.reshape(-1, 1))
    tfDF.insert(colCount, featureName, _tf, True)
    # tfDF[featureName] = _tf
  return tfDF

def dt_zScore(df, featureList):
  getColNames = df.columns
  tfDF = df[[getColNames[0], getColNames[1], getColNames[2], getColNames[3]]]
  colCount = 3
  for featureName in featureList:
    colCount = colCount+1
    colFeatDF = df[featureName]
    scaler = StandardScaler()
    _tf = scaler.fit_transform(colFeatDF.values.reshape(-1, 1))
    tfDF.insert(colCount, featureName, _tf, True)
    # tfDF[featureName] = _tf
  return tfDF

def dt_yeoJohnson(df, featureList):
  getColNames = df.columns
  tfDF = df[[getColNames[0], getColNames[1], getColNames[2], getColNames[3]]]
  colCount = 3
  for featureName in featureList:
    colCount = colCount+1
    colFeatDF = df[featureName]
    scaler = PowerTransformer(method='yeo-johnson')
    _tf = scaler.fit_transform(colFeatDF.values.reshape(-1, 1))
    tfDF.insert(colCount, featureName, _tf, True)
    # tfDF[featureName] = _tf
  return tfDF

def dt_yeoJohnson_minMax(df, featureList):
  _df_1 = dt_yeoJohnson(df, featureList)
  _df_2 = dt_minMax(_df_1, featureList)
  return _df_2

def dataClustering(cMethod):
  # print("Data clustering method: "+cMethod)
  if cMethod == "random_forest":
    return dc_randomForest()
  elif cMethod == "dbscan":
    return dc_dbscan()
  elif cMethod == "hdbscan":
    return dc_hdbscan()
  elif cMethod == "k_means":
    return dc_kMeans()
  else:
    print("ERROR: unavailable data clustering method selected")

def dc_randomForest():
  return 0

def dc_dbscan():
  return 0

def dc_hdbscan():
  return 0

def dc_kMeans():
  return 0

def dimensionReduction(drMethod, df, featureList):
  # print("Dimension reduction method: "+drMethod)
  if drMethod == "MDS":
    return dr_MDS(df, featureList)
  elif drMethod == "PCA":
    return dr_PCA(df, featureList)
  elif drMethod == "ICA":
    return dr_ICA(df, featureList)
  elif drMethod == "t_SNE":
    return dr_TSNE(df, featureList)
  elif drMethod == "PLS":
    return dr_PLS(df, featureList)
  else:
    print("ERROR: unavailable dimension reduction method selected")
    return df[['x', 'y']]

def dr_MDS(df, featureList):
  drm = MDS(n_components=2, random_state=0)
  drDF = drm.fit_transform(df[featureList])
  return drDF

def dr_PCA(df, featureList):
  drm = PCA(n_components=2, random_state=0)
  drDF = drm.fit_transform(df[featureList])
  return drDF

def dr_ICA(df, featureList):
  drm = FastICA(n_components=2, random_state=0)
  drDF = drm.fit_transform(df[featureList])
  return drDF

def dr_TSNE(df, featureList):
  drm = TSNE(learning_rate=100, random_state=0)
  drDF = drm.fit_transform(df[featureList])
  return drDF

def dr_PLS(df, featureList):
  drm = PLSRegression(n_components=2)
  drDF, _ = drm.fit_transform(df[featureList], df["label"])
  return drDF

FEATURE_ordered = ["intensity", "color", "orientation", "curvature", "center_bias", "entropy_rate", "log_spectrum", "HOG"]
DATA_TRANSFORMATION_LIST = ["min_max"]
DIMENSION_REDUCTION_LIST = ["MDS", "PCA"]
PATCH_SIZE = 20

FIX_DATA_DIR = "./fix/"
DATASET_LIST = os.listdir(FIX_DATA_DIR)

MID_FILE_GEN_PATH = ""
CACHE_FILE_GEN_PATH = ""
# midCacheFlag = True
ERROR_LOG = []
for dt in DATA_TRANSFORMATION_LIST:
    for dr in DIMENSION_REDUCTION_LIST:
        for dataName in DATASET_LIST:
            path = FIX_DATA_DIR + dataName +"/"
            STI_CLASS_LIST = os.listdir(path)
            for stiClass in STI_CLASS_LIST:
                path = FIX_DATA_DIR + dataName +"/"+ stiClass +"/"
                STI_DIR_LIST = os.listdir(path)
                for stiDirName in STI_DIR_LIST:
                    stiNameWithoutExe = ""
                    stiDirNameSplit = stiDirName.split("_")
                    if len(stiDirNameSplit) == 2:
                        stiNameWithoutExe = stiDirNameSplit[0]
                    else:
                        for i in range(0, len(stiDirNameSplit)):
                            if i==0:
                                stiNameWithoutExe = stiDirNameSplit[i]
                            elif i==len(stiDirNameSplit)-1:
                                break
                            else:
                                stiNameWithoutExe = stiNameWithoutExe +"_"+ stiDirNameSplit[i]
                    path = FIX_DATA_DIR + dataName +"/"+ stiClass +"/"+ stiDirName +"/"
                    FIX_FILE_LIST = os.listdir(path)
                    # MID_FILE_GEN_PATH = "./cache_mid/midcache-"+ dataName +"-"+ stiClass +"-"+ stiDirName +"-"+ dt +"-"+ dr +"-"+ str(len(FIX_FILE_LIST)) +".csv"
                    CACHE_FILE_GEN_PATH = "./cache/cache_"+ dataName +"-"+ stiClass +"-"+ stiDirName +"-"+ dt +"-"+ dr +"-"+ str(len(FIX_FILE_LIST)) +".csv"
                    if os.path.exists(CACHE_FILE_GEN_PATH):
                        print("CACHE: Already exists: "+CACHE_FILE_GEN_PATH)
                        continue
                    # midCacheFag = True
                    # if os.path.exists(MID_FILE_GEN_PATH):
                    #     print("MID: Already exists: "+MID_FILE_GEN_PATH)
                    #     midCacheFlag = False
                    #     break
                    print(CACHE_FILE_GEN_PATH)
                    # print("start time: "+datetime.today().strftime("%Y/%m/%d-%H:%M:%S"))
                    gtFixMapPath = "./ground_truth/" + dataName +"/"+ stiClass +"/"+ stiNameWithoutExe +".jpg"
                    groundTruthFixMap = cv2.imread(gtFixMapPath)
                    fmHeight, fmWidth = groundTruthFixMap.shape[:2]
                    loadedFeatureDataMatrix = []
                    for _f in FEATURE_ordered:
                        featureFilePath = "./feature/" + _f +"/"+ dataName +"/"+ stiClass +"/"+ stiNameWithoutExe +".csv"
                        featureDF = pd.read_csv(featureFilePath)
                        loadedFeatureDataMatrix.append(featureDF)
                    aggregatedDataList = []
                    for fixFileName in FIX_FILE_LIST:
                        path = FIX_DATA_DIR + dataName +"/"+ stiClass +"/"+ stiDirName +"/"+ fixFileName
                        df = pd.read_csv(path, header=None)
                        fixList = df.values.tolist()
                        observer = dataName +"/"+ stiClass +"/"+ stiDirName +"/"+ fixFileName
                        for _fp in fixList:
                            _x = int(_fp[0])
                            _y = int(_fp[1])
                            _label = label_groundTruthFixationMap(groundTruthFixMap, _x, _y)
                            _midStack = [observer, _x, _y, _label]
                            for i in range(0, len(FEATURE_ordered)):
                                fMean = getFeatureMeanVal(loadedFeatureDataMatrix[i], _x, _y, fmWidth, fmHeight, PATCH_SIZE)
                                _midStack.append(fMean)
                            aggregatedDataList.append(_midStack)
                    aggDF = []
                    dfCols = ["id", "x", "y", "label"]
                    for featName in FEATURE_ordered:
                      dfCols.append(featName)
                    # if midCacheFlag == True:
                    #   print("make file: "+MID_FILE_GEN_PATH)
                    #   aggDF = pd.DataFrame(aggregatedDataList, columns=dfCols)
                    #   aggDF.to_csv(MID_FILE_GEN_PATH, mode='w', index=False)
                    # else:
                    #   print("load file: "+MID_FILE_GEN_PATH)
                    #   aggDF = pd.read_csv(MID_FILE_GEN_PATH)
                    aggDF = pd.DataFrame(aggregatedDataList, columns=dfCols)
                    tfDF = dataTransformation(dt, aggDF, FEATURE_ordered)
                    drRes = dimensionReduction(dr, tfDF, FEATURE_ordered)
                    drDF = pd.DataFrame(drRes, columns=['x', 'y'])

                    indexCount = 0
                    processedDF = pd.DataFrame(aggDF['id'].values.tolist(), columns=['id'])
                    indexCount = indexCount+1
                    processedDF.insert(indexCount, "x", drDF['x'].values.tolist(), True)
                    indexCount = indexCount+1
                    processedDF.insert(indexCount, "y", drDF['y'].values.tolist(), True)
                    indexCount = indexCount+1
                    processedDF.insert(indexCount, "label", aggDF['label'].values.tolist(), True)
                    indexCount = indexCount+1
                    for featName in FEATURE_ordered:
                      processedDF.insert(indexCount, featName, tfDF[featName].values.tolist(), True)
                      indexCount = indexCount+1
                    
                    dataColumns = processedDF.columns.values.tolist()
                    processedDataList = processedDF.values.tolist()
                    rawDataList = aggDF.values.tolist()
                    processedDF.to_csv(CACHE_FILE_GEN_PATH, mode='w', index=False, header=True)

                
                        