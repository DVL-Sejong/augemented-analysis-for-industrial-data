import os
import pandas as pd
import numpy as np

RAW_DIR = "./eye tracking data (Ehinger)"
SEG_DIR = "./segmentation_Ehinger"
OUT_DIR = "./object_fixation_raw/Ehinger"
raw_list = os.listdir(RAW_DIR)

for fileName in raw_list:
    # print(fileName)
    raw_file_path = RAW_DIR+"/"+fileName
    sti_name = fileName.split('.')[0].split('_')[0]+"_"+fileName.split('.')[0].split('_')[1]
    seg_file_path = SEG_DIR+"/"+sti_name+".csv"
    seg_df = pd.read_csv(seg_file_path, index_col=False, header=None)
    raw_df = pd.read_csv(raw_file_path)
    raw_df = raw_df.dropna()

    label_list = []
    for index, row in raw_df.iterrows():
        x = float(row[0])
        y = float(row[1])
        l = int(row[2])
        if x < 0 or y < 0 or x >= 800 or y >= 600:
            continue
        if l != 0:
            continue
        obj_l = int(seg_df[int(x)][int(y)])
        label_list.append([x, y, obj_l])
    out_df = pd.DataFrame(label_list, columns=["x", "y", "object"])
    out_file_path = OUT_DIR+"/"+fileName
    out_df.to_csv(out_file_path, index=False)
    print(out_file_path)