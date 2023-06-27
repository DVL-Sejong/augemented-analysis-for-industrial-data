import sys
import os
import numpy as np
import cv2

# from scipy.misc import imresize
from scipy.stats import entropy


def NSS(saliency_map, ground_truth_map):
  """"
  normalized scanpath saliency between two different
  saliency maps as the mean value of the normalized saliency map at
  fixation locations.
    Computer NSS score.
    :param saliency_map : predicted saliency map
    :param fixation_map : ground truth saliency map.
    :return score: float : score
  """
  if not isinstance(saliency_map, np.ndarray):
    saliency_map = np.array(saliency_map)

  if not isinstance(ground_truth_map, np.ndarray):
    ground_truth_map = np.array(ground_truth_map)

  if saliency_map.size != ground_truth_map.size:
    saliency_map = cv2.resize(saliency_map, dsize=(ground_truth_map.shape[1], ground_truth_map.shape[0]))
    # saliency_map = imresize(saliency_map, fixation_map.shape)

  MAP = (saliency_map - saliency_map.mean()) / (saliency_map.std())
  mask = ground_truth_map.astype(np.bool)
  score = MAP[mask].mean()
  return score

def CC(saliency_map, ground_truth_map):
  """
  This finds the linear correlation coefficient between two different
  saliency maps (also called Pearson's linear coefficient).
  score=1 or -1 means the maps are correlated
  score=0 means the maps are completely uncorrelated
  saliencyMap1 and saliencyMap2 are 2 real-valued matrices
    Computer CC score .
    :param saliency_map : first saliency map
    :param saliency_map_gt : second  saliency map.
    :return score: float : score
  """
  if not isinstance(saliency_map, np.ndarray):
    saliency_map = np.array(saliency_map, dtype=np.float32)
  elif saliency_map.dtype != np.float32:
    saliency_map = saliency_map.astype(np.float32)

  if not isinstance(ground_truth_map, np.ndarray):
    ground_truth_map = np.array(ground_truth_map, dtype=np.float32)
  elif saliency_map.dtype != np.float32:
    ground_truth_map = ground_truth_map.astype(np.float32)

  if saliency_map.size != ground_truth_map.size:
    saliency_map = cv2.resize(saliency_map, dsize=(ground_truth_map.shape[1], ground_truth_map.shape[0]))
    # saliency_map = imresize(saliency_map, ground_truth_map.shape)

  saliency_map = (saliency_map - saliency_map.mean()) / (saliency_map.std())
  ground_truth_map = (ground_truth_map - ground_truth_map.mean()) / (ground_truth_map.std())
  score = np.corrcoef(saliency_map.flatten(),ground_truth_map.flatten())[0][1]
  return score

def KLdiv(saliency_map, ground_truth_map):
  """
  This finds the KL-divergence between two different saliency maps when
  viewed as distributions: it is a non-symmetric measure of the information
  lost when saliencyMap is used to estimate fixationMap.
    Computer KL-divergence.
    :param saliency_map : predicted saliency map
    :param fixation_map : ground truth saliency map.
    :return score: float : score
  """
  if saliency_map.size != ground_truth_map.size:
    saliency_map = cv2.resize(saliency_map, dsize=(ground_truth_map.shape[1], ground_truth_map.shape[0]))

  if not isinstance(saliency_map, np.ndarray):
    saliency_map = np.array(saliency_map, dtype=np.float32)
  elif saliency_map.dtype != np.float32:
    saliency_map = saliency_map.astype(np.float32)

  if not isinstance(ground_truth_map, np.ndarray):
    ground_truth_map = np.array(ground_truth_map, dtype=np.float32)
  elif ground_truth_map.dtype != np.float32:
    ground_truth_map = ground_truth_map.astype(np.float32)

  EPS = np.finfo(np.float32).eps
  # the function will normalize maps before computing Kld
  score = entropy(saliency_map.flatten() + EPS, ground_truth_map.flatten() + EPS)
  return score

def AUC(saliency_map, ground_truth_map):
  """Computes AUC for given saliency map 'saliency_map' and given
  fixation map 'fixation_map'
  """
  def area_under_curve(predicted, actual, labelset):
    def roc_curve(predicted, actual, cls):
      si = np.argsort(-predicted)
      tp = np.cumsum(np.single(actual[si]==cls))
      fp = np.cumsum(np.single(actual[si]!=cls))
      tp = tp/np.sum(actual==cls)
      fp = fp/np.sum(actual!=cls)
      tp = np.hstack((0.0, tp, 1.0))
      fp = np.hstack((0.0, fp, 1.0))
      return tp, fp
    def auc_from_roc(tp, fp):
      h = np.diff(fp)
      auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
      return auc

    tp, fp = roc_curve(predicted, actual, np.max(labelset))
    auc = auc_from_roc(tp, fp)
    return auc

  ground_truth_map = (ground_truth_map>0.7).astype(int)
  salShape = saliency_map.shape
  fixShape = ground_truth_map.shape

  predicted = saliency_map.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
  actual = ground_truth_map.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
  labelset = np.arange(2)
  return area_under_curve(predicted, actual, labelset)

def SAUC(saliency_map, ground_truth_map, shuf_map=np.zeros((480,640)), step_size=.01):
  # shuf_map=np.zeros(ground_truth_map.shape)
  # shuf_map = ground_truth_map
  """
    please cite:  https://github.com/NUS-VIP/salicon-evaluation
    calculates shuffled-AUC score.
    :param salinecy_map : predicted saliency map
    :param fixation_map : ground truth saliency map.
    :return score: int : score
  """
  
  saliency_map -= np.min(saliency_map)
  ground_truth_map = np.vstack(np.where(ground_truth_map!=0)).T
  
  if np.max(saliency_map) > 0:
    saliency_map = saliency_map / np.max(saliency_map)
  Sth = np.asarray([ saliency_map[y-1][x-1] for y,x in ground_truth_map ])
  
  Nfixations = len(ground_truth_map)
  others = np.copy(shuf_map)
  for y,x in ground_truth_map:
    others[y-1][x-1] = 0

  ind = np.nonzero(others) # find fixation locations on other images
  nFix = shuf_map[ind]
  randfix = saliency_map[ind]
  Nothers = sum(nFix)

  allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
  allthreshes = allthreshes[::-1]
  tp = np.zeros(len(allthreshes)+2)
  fp = np.zeros(len(allthreshes)+2)
  tp[-1]=1.0
  fp[-1]=1.0
  tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
  fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]
  score = np.trapz(tp,fp)
  return score

def IG(saliency_map, ground_truth_map, baseline_map=np.zeros((480,640))):
  """
    please cite:
    calculates Information gain score.
    :param salinecy_map : predicted saliency map
    :param fixation_map : ground truth saliency map.
    :param baseline_fixation_map : a baseline fixtion map
    :return score: int : score
  """
  if saliency_map.size != ground_truth_map.size:
    saliency_map = cv2.resize(saliency_map, dsize=(ground_truth_map.shape[1], ground_truth_map.shape[0]))

  if not isinstance(saliency_map, np.ndarray):
    saliency_map = np.array(saliency_map, dtype=np.float32)
  elif saliency_map.dtype != np.float32:
    saliency_map = saliency_map.astype(np.float32)

  if not isinstance(ground_truth_map, np.ndarray):
    ground_truth_map = np.array(ground_truth_map, dtype=np.float32)
  elif ground_truth_map.dtype != np.float32:
    ground_truth_map = ground_truth_map.astype(np.float32)

  if not isinstance(baseline_map, np.ndarray):
    baseline_map = np.array(baseline_map, dtype=np.float32)
  elif ground_truth_map.dtype != np.float32:
    baseline_map = baseline_map.astype(np.float32)

  saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
  saliency_map = saliency_map / saliency_map.sum()
  baseline_map = (baseline_map - baseline_map.min()) / (baseline_map.max() - baseline_map.min())
  baseline_map = baseline_map / baseline_map.sum()
  fixs = ground_truth_map.astype(np.bool)
  EPS = np.finfo(np.float32).eps
  return (np.log2(EPS + saliency_map[fixs]) - np.log2(EPS + baseline_map[fixs])).mean()

def SIM(saliency_map, ground_truth_map):
  """
    Compute similarity score.
    :param saliency_map : predicted saliency map
    :param fixation_map : ground truth saliency map.
    :return score: float : score
  """
  if saliency_map.size != ground_truth_map.size:
    saliency_map = cv2.resize(saliency_map, dsize=(ground_truth_map.shape[1], ground_truth_map.shape[0]))
    
  if not isinstance(saliency_map, np.ndarray):
    saliency_map = np.array(saliency_map, dtype=np.float32)
  elif saliency_map.dtype != np.float32:
    saliency_map = saliency_map.astype(np.float32)

  if not isinstance(ground_truth_map, np.ndarray):
    ground_truth_map = np.array(ground_truth_map, dtype=np.float32)
  elif ground_truth_map.dtype != np.float32:
    ground_truth_map = ground_truth_map.astype(np.float32)

  saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
  saliency_map = saliency_map / saliency_map.sum()
  ground_truth_map = (ground_truth_map - ground_truth_map.min()) / (ground_truth_map.max() - ground_truth_map.min())
  ground_truth_map = ground_truth_map / ground_truth_map.sum()
  return np.minimum(saliency_map, ground_truth_map).sum()


def normalize_map(s_map):
  # normalize the salience map (as done in MIT code)
  norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
  return norm_s_map

# def discretize_gt(gt):
#   import warnings
#   warnings.warn('can improve the way GT is discretized')
#   return gt/255

# def auc_shuff(s_map, gt, other_map, splits=100, step_size=0.1):
#   gt = discretize_gt(gt)
#   other_map = discretize_gt(other_map)
#   num_fixations = np.sum(gt)
#   x,y = np.where(other_map==1)
#   other_map_fixs = []
#   for j in zip(x,y):
#     other_map_fixs.append(j[0]*other_map.shape[0] + j[1])
#   ind = len(other_map_fixs)
#   assert ind==np.sum(other_map), 'something is wrong in auc shuffle'

#   num_fixations_other = min(ind,num_fixations)
#   num_pixels = s_map.shape[0]*s_map.shape[1]
#   random_numbers = []
#   for i in range(0,splits):
#     temp_list = []
#     t1 = np.random.permutation(ind)
#     for k in t1:
#       temp_list.append(other_map_fixs[k])
#     random_numbers.append(temp_list)

#   aucs = []
#   # for each split, calculate auc
#   for i in random_numbers:
#     r_sal_map = []
#     for k in i:
#       r_sal_map.append(s_map[k%s_map.shape[0]-1, k/s_map.shape[0]])
#     # in these values, we need to find thresholds and calculate auc
#     thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#     r_sal_map = np.array(r_sal_map)

#     # once threshs are got
#     thresholds = sorted(set(thresholds))
#     area = []
#     area.append((0.0,0.0))
#     for thresh in thresholds:
#       # in the salience map, keep only those pixels with values above threshold
#       temp = np.zeros(s_map.shape)
#       temp[s_map>=thresh] = 1.0
#       num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
#       tp = num_overlap/(num_fixations*1.0)

#       #fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
#       # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
#       fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

#       area.append((round(tp,4),round(fp,4)))

#     area.append((1.0,1.0))
#     area.sort(key = lambda x:x[0])
#     tp_list =  [x[0] for x in area]
#     fp_list =  [x[1] for x in area]

#     aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))

#   return np.mean(aucs)  



# this is just the name its not actually discretised (binary)
gt = cv2.imread('001_gt.jpg',0)
s_map = cv2.imread('001_SaliencyMap.jpg',0)
s_map_norm = normalize_map(s_map)
gt_resize = cv2.resize(gt, dsize=(s_map_norm.shape[1],s_map_norm.shape[0]))
# print("gt.shape")
# print(gt.shape)

# print("s_map.shape")
# print(s_map.shape)

# print("s_map_norm.shape")
# print(s_map_norm.shape)

# sHeight, sWidth = s_map.shape[:2]
# gt = cv2.resize(gt_origin, dsize=(sWidth,sHeight))
# print(gt.shape)
# s_map_norm = normalize_map(s_map)

# NSS score
# score_NSS = NSS(s_map, gt)
# print("score_NSS: "+str(score_NSS))
# score_CC = CC(s_map_norm, gt)
# print("score_CC: "+str(score_CC))
# score_KLdiv = KLdiv(s_map_norm, gt)
# print("score_KLdiv: "+str(score_KLdiv))
# score_AUC = AUC(s_map_norm, gt)
# print("score_AUC: "+str(score_AUC))
# score_SIM = SIM(s_map_norm, gt)
# print("score_SIM: "+str(score_SIM))

score_SAUC_1 = SAUC(s_map_norm, gt_resize, gt_resize)
print("score_SAUC: "+str(score_SAUC_1))
score_IG = IG(s_map_norm, gt_resize, gt_resize)
print("score_IG: "+str(score_IG))
