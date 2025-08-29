import os
import pandas as pd
import numpy as np

DATASET = "collected-x2-30"
RAW_DIR = "./eye tracking data (collected)"
SEG_DIR = "./segmentation_"+DATASET
OUT_DIR = "./object_fixation_raw_all_events/"+DATASET
raw_list = os.listdir(RAW_DIR)

for fileName in raw_list:
    # print(fileName)
    raw_file_path = RAW_DIR+"/"+fileName
    sti_name = fileName.split('.')[0].split('__')[0].split('+')[1]
    user_name = fileName.split('.')[0].split('__')[1]
    seg_file_path = SEG_DIR+"/"+sti_name+".csv"
    if os.path.isfile(seg_file_path) == False:
        print("No segmentation data")
        continue
    seg_df = pd.read_csv(seg_file_path, index_col=False, header=None)
    # print(seg_df)

    raw_df = pd.read_csv(raw_file_path)
    raw_df = raw_df.dropna()

    STI_WIDTH = len(seg_df.columns)
    STI_HEIGHT = len(seg_df)

    label_list = []
    for index, row in raw_df.iterrows():
        x = float(row['x'])
        y = float(row['y'])
        if x < 0 or y < 0 or x >= STI_WIDTH or y >= STI_HEIGHT:
            continue
        obj_l = int(seg_df[int(x)][int(y)])
        label_list.append([x, y, obj_l])
    out_df = pd.DataFrame(label_list, columns=["x", "y", "object"])
    out_file_path = OUT_DIR+"/"+sti_name+"__"+user_name+".csv"
    out_df.to_csv(out_file_path, index=False)
    print(out_file_path)