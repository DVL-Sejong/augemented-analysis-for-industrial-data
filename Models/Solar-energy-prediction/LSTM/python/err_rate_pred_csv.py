import os
import pandas as pd

ONM_ID = "03"
MONTH = 5

# 발전량 데이터 로드
file_path = "./data_"+ONM_ID+"/2022"+str(MONTH).zfill(2)+".csv"
real_df = pd.read_csv(file_path, index_col=False)

# LR_LIST = [0.005, 0.0001, 0.0005, 1e-05, 5e-06, 5e-06, 5e-07]

# 예측치 데이터 로드
pred_dir = "./300_5e-07/"
print(pred_dir)
pred_file_path = pred_dir+"2022"+str(MONTH).zfill(2)+".csv"
pred_df = pd.read_csv(pred_file_path, index_col=False)

MODEL = ['model1', 'model2']
for _model_type in MODEL:
    sum_err_rate = 0
    month_err_rate_list = []
    for index, row in real_df.iterrows():
        date = row['TIME']
        # print(date)
        hours_cols = ["HR_01", "HR_02", "HR_03", "HR_04", "HR_05", "HR_06", "HR_07", "HR_08", "HR_09", "HR_10", "HR_11", "HR_12", "HR_13", "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23", "HR_24"]
        capacity = float(row['CAPACITY'])
        over_10_per_list = []
        for _h in hours_cols:
            val = float(row[_h])
            if val >= capacity/10:
                over_10_per_list.append([int(_h.split('_')[1])-1, val])

        # initialize
        pred_list = []
        for _hv in over_10_per_list:
            is_date = pred_df['date'] == date
            is_time_step = pred_df['time_step'] == _hv[0]
            is_model = pred_df['model'] == _model_type
            is_predType = pred_df['predType'] == 1
            pred_val_1 = pred_df[is_date & is_time_step & is_predType & is_model]['pred_val'].values.tolist()
            if len(pred_val_1) == 0:
                continue
            is_predType = pred_df['predType'] == 2
            pred_val_2 = pred_df[is_date & is_time_step & is_predType & is_model]['pred_val'].values.tolist()
            if len(pred_val_2) == 0:
                continue
            pred_list.append([_hv[0], pred_val_1[0], pred_val_2[0]])
        
        for i in range(len(pred_list)):
            real = float(over_10_per_list[i][1])
            pred_1 = float(pred_list[i][1])
            pred_2 = float(pred_list[i][2])
            diff_1 = abs(real-pred_1)
            diff_2 = abs(real-pred_2)
            err_rate_1 = diff_1/capacity*100
            err_rate_2 = diff_2/capacity*100
            err_rate = (err_rate_1+err_rate_2)/2
            sum_err_rate = sum_err_rate+err_rate
            month_err_rate_list.append([date, _model_type, err_rate])

    if len(month_err_rate_list) != 0:
        sum_err_rate = sum_err_rate/len(month_err_rate_list)
        print("MODEL: ", _model_type, ", error rate: ", sum_err_rate)
        err_df = pd.DataFrame(month_err_rate_list, columns=['date', 'model', 'error'])
        err_df.to_csv(pred_dir+_model_type+"_err_"+str(MONTH).zfill(2)+".csv", index=False)


# import os
# import pandas as pd

# MONTH = 5

# # 발전량 데이터 로드
# file_path = "./data_03/2022"+str(MONTH).zfill(2)+".csv"
# real_df = pd.read_csv(file_path, index_col=False)

# # LR_LIST = [0.005, 0.0001, 0.0005, 1e-05, 5e-06, 5e-06, 5e-07]

# # 예측치 데이터 로드
# pred_dir = "./300_5e-07/"
# print(pred_dir)
# pred_file_path = pred_dir+"2022"+str(MONTH).zfill(2)+".csv"
# pred_df = pd.read_csv(pred_file_path, index_col=False)

# MODEL = ['model1', 'model2']
# for _model_type in MODEL:
#     sum_err_rate = 0
#     month_err_rate_list = []
#     for index, row in real_df.iterrows():
#         date = row['TIME']
#         # print(date)
#         hours_cols = ["HR_01", "HR_02", "HR_03", "HR_04", "HR_05", "HR_06", "HR_07", "HR_08", "HR_09", "HR_10", "HR_11", "HR_12", "HR_13", "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23", "HR_24"]
#         capacity = float(row['CAPACITY'])
#         over_10_per_list = []
#         for _h in hours_cols:
#             val = float(row[_h])
#             if val >= capacity/10:
#                 over_10_per_list.append([int(_h.split('_')[1])-1, val])

#         # initialize
#         pred_list = []
#         for _hv in over_10_per_list:
#             is_date = pred_df['date'] == date
#             is_time_step = pred_df['time_step'] == _hv[0]
#             is_model = pred_df['model'] == _model_type
#             is_predType = pred_df['predType'] == 1
#             pred_val_1 = pred_df[is_date & is_time_step & is_predType & is_model]['pred_val'].values.tolist()
#             if len(pred_val_1) == 0:
#                 continue
#             pred_list.append([_hv[0], pred_val_1[0]])
        
#         for i in range(len(pred_list)):
#             real = float(over_10_per_list[i][1])
#             pred_1 = float(pred_list[i][1])
#             diff_1 = abs(real-pred_1)
#             err_rate_1 = diff_1/capacity*100
#             sum_err_rate = sum_err_rate+err_rate_1
#             month_err_rate_list.append([date, _model_type, err_rate_1])

#     if len(month_err_rate_list) != 0:
#         sum_err_rate = sum_err_rate/len(month_err_rate_list)
#         print("MODEL: ", _model_type, ", error rate: ", sum_err_rate)
#         err_df = pd.DataFrame(month_err_rate_list, columns=['date', 'model', 'error'])
#         err_df.to_csv(pred_dir+_model_type+"_err_"+str(MONTH).zfill(2)+".csv", index=False)
