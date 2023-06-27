import os
import csv

def create_dir(_dirpath):
    try:
        if not os.path.exists(_dirpath):
            os.makedirs(_dirpath)
    except OSError:
        print("Error: creating directory."+_dirpath)

def csvWriter(_outputfilename, _data):
    f = open(_outputfilename, 'w', newline='', encoding='utf-8')
    c = csv.writer(f)
    for _row in _data:
        c.writerow(_row)
    f.close()

def csvReader(_filename):
    _data = []
    f = open(_filename, 'r', encoding='utf-8')
    c = csv.reader(f)
    for _row in c:
        _data.append(_row)
    return _data

IMG_WIDTH = 800
IMG_HEIGHT = 600

STIMULUS = "U0121_1RTE"
USER = "0"
GAZE_CSV = STIMULUS+"_"+USER+".csv"
SALIENCY_COL_CSV = STIMULUS+"_color"+".csv"
SALIENCY_INT_CSV = STIMULUS+"_intensity"+".csv"
SALIENCY_ORI_CSV = STIMULUS+"_orientation"+".csv"

GAZE_DATA = csvReader(GAZE_CSV)
SALIENCY_COL_DATA = csvReader(SALIENCY_COL_CSV)
SALIENCY_INT_DATA = csvReader(SALIENCY_INT_CSV)
SALIENCY_ORI_DATA = csvReader(SALIENCY_ORI_CSV)

FEATURE_DATA = SALIENCY_ORI_DATA

divLevel = 0.1
minVal = 0
maxVal = 1
arrLen = 10

COUNTING = []
for i in range(0, arrLen):
    COUNTING.append(0)

for _row in FEATURE_DATA:
    for _col in _row:
        idx = round(float(_col)/divLevel)
        if idx >= arrLen:
            idx -= 1
        if idx < 0:
            idx = 0
        COUNTING[idx] += 1

print(COUNTING)

# intensity
# color
# orientation
OUT_FILE = STIMULUS+"_saliency_"+"orientation"+".csv"
OUT = []
OUT.append(["level", "value"])
for i in range(0, len(COUNTING)):
    OUT.append([i, COUNTING[i]])

csvWriter(OUT_FILE, OUT)