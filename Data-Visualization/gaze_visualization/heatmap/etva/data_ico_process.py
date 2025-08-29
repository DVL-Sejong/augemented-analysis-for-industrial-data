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

IDX_GAZE_DATA = []
overIDXcount = 0
for _row in GAZE_DATA:
    converted_x = round(float(_row[0]))
    converted_y = round(float(_row[1]))
    if converted_x >= IMG_WIDTH or converted_y >= IMG_HEIGHT:
        overIDXcount += 1
        continue
    _c = [converted_x, converted_y]
    IDX_GAZE_DATA.append(_c)

print("Counting out of range: ")
print(overIDXcount)

OUT_DATA = []
for _data in IDX_GAZE_DATA:
    _c = _data[0]
    _r = _data[1]
    _sc = SALIENCY_COL_DATA[_r][_c]
    _si = SALIENCY_INT_DATA[_r][_c]
    _so = SALIENCY_ORI_DATA[_r][_c]

    _row = [_c, _r, _sc, _si, _so]
    OUT_DATA.append(_row)

OUT_FILE = "out_data.csv"
csvWriter(OUT_FILE, OUT_DATA)