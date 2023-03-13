import os
import pandas as pd

ID = "04"
FAC_DATA_DIR = "../data_id_"+ID+"/"
FAC_DATA_LIST = os.listdir(FAC_DATA_DIR)

data_list = []
prev_m = 0
for file_name in FAC_DATA_LIST:
    month = int(file_name[4:6])
    if prev_m == 0:
        prev_m = month
    
    if prev_m != month:
        print(prev_m)
        solar_df = pd.DataFrame(data_list, columns=["TIME", "CAPACITY", "CBP_GEN_ID", "COMPANY_NM", "GEN_ID", "GEN_NM", "HR_01", "HR_02", "HR_03", "HR_04", "HR_05", "HR_06", "HR_07", "HR_08", "HR_09", "HR_10", "HR_11", "HR_12", "HR_13", "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23", "HR_24", "PATN_DT", "SET_GEN_NM"])
        solar_df.to_csv("./data_"+ID+"/2022"+str(prev_m).zfill(2)+".csv", index=False, encoding='utf-8')
        data_list = []
    df = pd.read_csv(FAC_DATA_DIR+file_name, index_col=False)
    df_list = df.values.tolist()
    data_list.append(df_list[0])
    prev_m = month
print(prev_m)
solar_df = pd.DataFrame(data_list, columns=["TIME", "CAPACITY", "CBP_GEN_ID", "COMPANY_NM", "GEN_ID", "GEN_NM", "HR_01", "HR_02", "HR_03", "HR_04", "HR_05", "HR_06", "HR_07", "HR_08", "HR_09", "HR_10", "HR_11", "HR_12", "HR_13", "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23", "HR_24", "PATN_DT", "SET_GEN_NM"])
solar_df.to_csv("./data_"+ID+"/2022"+str(prev_m).zfill(2)+".csv", index=False, encoding='utf-8')
data_list = []