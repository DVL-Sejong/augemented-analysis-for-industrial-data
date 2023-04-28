import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


x = np.array([[0, 2], [2, 6], [11, 4], [9, 9]])
y = np.array([[3, 2], [5, 6], [14, 4], [12, 9]])
z = np.array([[2, 2], [0, 8], [9, 3], [9, 11]])
