import os
import pandas as pd

ID = 3
print("ID:",str(ID))

DATA_DIR = "./data_"+str(ID).zfill(2)+"/"
DATA_LIST = os.listdir(DATA_DIR)

# model1 = LSTM 모델, model2 = 앙상블 모델
MODELS = ['model1', 'model2']
# 모델 선택
# model = MODELS[1]
for model in MODELS:
    month_err_rate_list = []
    for file_name in DATA_LIST:
        # print("Month: "+file_name.split('.')[0])
        err_rate_res_list = []
        file_path = DATA_DIR+file_name
        df = pd.read_csv(file_path, index_col=False)

        for index, row in df.iterrows():
            date = row['TIME']
            # print(date)
            hours_cols = ["HR_01", "HR_02", "HR_03", "HR_04", "HR_05", "HR_06", "HR_07", "HR_08", "HR_09", "HR_10", "HR_11", "HR_12", "HR_13", "HR_14", "HR_15", "HR_16", "HR_17", "HR_18", "HR_19", "HR_20", "HR_21", "HR_22", "HR_23", "HR_24"]
            capacity = float(row['CAPACITY'])
            over_10_per_list = []
            for _h in hours_cols:
                val = float(row[_h])
                if val >= capacity/10:
                    over_10_per_list.append([int(_h.split('_')[1])-1, val])

            # sum_ICM = 0
            # for _v in over_10_per_list:
            #     sum_ICM = sum_ICM + _v[1]

            # initialize
            pred_list = []
            for _hv in over_10_per_list:
                pred_list.append([0, 0])

            # 1차 예측치 조회 쿼리 생성
            date_q = "date="+str(date)
            rsrsId_q = "rsrsId="+str(ID)
            predType_q = "predType="+"1"
            request_json_qry = "http://203.250.148.62:8000/getPredResult?"+date_q+"&"+rsrsId_q+"&"+predType_q
            try:
                pred_1_df = pd.read_json(request_json_qry)
                # 1차 예측치 리스트에 저장
                idx = 0
                for _hv in over_10_per_list:
                    time = str(_hv[0])
                    pred_val = pred_1_df['errRate'][model][time]
                    pred_list[idx][0] = pred_val
                    idx = idx+1

                # 2차 예측치 조회 쿼리 생성
                predType_q = "predType="+"2"
                request_json_qry = "http://203.250.148.62:8000/getPredResult?"+date_q+"&"+rsrsId_q+"&"+predType_q
                pred_2_df = pd.read_json(request_json_qry)
                # 2차 예측치 리스트에 저장
                idx = 0
                for _hv in over_10_per_list:
                    time = str(_hv[0])
                    pred_val = pred_2_df['errRate'][model][time]
                    pred_list[idx][1] = pred_val
                    idx = idx+1

                # 예측 오차율 계산
                # sum_err_rate = 0
                # for i in range(0, len(pred_list)):
                #     pred_1 = pred_list[i][0]
                #     pred_2 = pred_list[i][1]
                #     real = over_10_per_list[i][1]
                #     diff_1 = abs(pred_1-real)
                #     diff_2 = abs(pred_2-real)
                #     err_rate_1 = diff_1/capacity*100
                #     err_rate_2 = diff_2/capacity*100
                #     err_rate = (err_rate_1+err_rate_2)/2
                #     sum_err_rate = sum_err_rate+err_rate
                sum_err_rate = 0
                for i in range(0, len(pred_list)):
                    pred_1 = pred_list[i][0]
                    pred_2 = pred_list[i][1]
                    err_rate = (pred_1+pred_2)/2
                    sum_err_rate = sum_err_rate+err_rate
                if len(pred_list) != 0:
                    avg_err_rate = sum_err_rate/len(pred_list)
                    err_rate_res_list.append([date, avg_err_rate])
            except:
                # print("EXCEPTION CONTROL - NO DATA: "+str(date))
                continue
        sum_err_rate_date = 0
        for _e in err_rate_res_list:
            sum_err_rate_date = sum_err_rate_date+_e[1]
        if len(err_rate_res_list) != 0:
            avg_err_rate_day = sum_err_rate_date/len(err_rate_res_list)
            err_rate_res_list.append(["average", avg_err_rate_day])
            month_err_rate_list.append([file_name.split('.')[0], avg_err_rate_day])

        res_df = pd.DataFrame(err_rate_res_list, columns=['date', 'error_rate'])
        res_df.to_csv("./err_"+str(ID).zfill(2)+"/"+model+"_"+str(ID).zfill(2)+"_"+file_name, index=False)

    month_df = pd.DataFrame(month_err_rate_list, columns=['date', 'error_rate'])
    print("MODEL: "+model)
    print(month_df)