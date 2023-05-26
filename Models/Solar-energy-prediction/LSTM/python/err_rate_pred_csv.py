import os
import pandas as pd

# 예측 월 설정
MONTH = 11
print("Target year and month: 2022, "+str(MONTH).zfill(2))
# Model parameters
EPOCH = 1000
LEARNING_LATE = 0.00001
SHIFT_DAYS = 16
BATCH_SIZE = 32

# 설비용량 로드
CAPACITY = 100

# 발전량 데이터 로드
file_path = "./green_inverter.csv"


real_load_df = pd.read_csv(file_path, index_col=False)
real_list = []
REAL_COLS = ["year", "month", "day", "hour", "minute", "date", "POWER", "over"]
for index, row in real_load_df.iterrows():
    year = int(row['time'].split(' ')[0].split('-')[0])
    month = int(row['time'].split(' ')[0].split('-')[1])
    day = int(row['time'].split(' ')[0].split('-')[2])
    hour = int(row['time'].split(' ')[1].split(':')[0])
    minute = int(row['time'].split(' ')[1].split(':')[1])
    date = str(year)+str(month).zfill(2)+str(day).zfill(2)

    over = True
    power = float(row['POWER'])
    if power < CAPACITY/10:
        # 설비용량 10프로 미만 시간대 예측 대상에서 제외
        over = False

    if month == MONTH:
        _row = [year, month, day, hour, minute, date, row['POWER'], over]
        real_list.append(_row)

real_df = pd.DataFrame(real_list, columns=REAL_COLS)
# print(real_df)

# 예측치 데이터 로드
pred_dir = "./pred_data/sensor/"
pred_file_path = pred_dir+str(EPOCH)+"_"+str(LEARNING_LATE)+"_{}_{}/".format(SHIFT_DAYS,BATCH_SIZE)+str(MONTH).zfill(2)+".csv"
pred_df = pd.read_csv(pred_file_path, index_col=False)


# model1 = LSTM 모델, model2 = 앙상블 모델
MODEL = ['model1', 'model2']
stack_err_rate_list = []
montly_model_err_list = []
for _model_type in MODEL:
    sum_err_rate = 0
    model_list = []
    err_rate_list = []
    
    for index, row in real_df.iterrows():
        over = row['over']
        if over == False:
            # 설비용량 10프로 미만 필터링
            continue
        date = row['date']
        year = row['year']
        month = row['month']
        day = row['day']
        hour = row['hour']
        power = row['POWER']
        
        # 1차 예측치 필터링
        is_date = pred_df['date'] == int(date)
        is_model = pred_df['model'] == _model_type
        is_time_step = pred_df['time_step'] == int(day)
        is_predType = pred_df['predType'] == 1

        # 1차 예측치 조회
        pred_val_1 = pred_df[is_date & is_time_step & is_predType & is_model]['pred_val'].values.tolist()
        if len(pred_val_1) == 0:
            # 누락 데이터 예외 처리
            continue

        # 2차 예측치 필터링
        is_date = pred_df['date'] == int(date)
        is_model = pred_df['model'] == _model_type
        is_time_step = pred_df['time_step'] == int(day)
        is_predType = pred_df['predType'] == 2
        pred_val_2 = pred_df[is_date & is_time_step & is_predType & is_model]['pred_val'].values.tolist()
        if len(pred_val_2) == 0:
            # 누락 데이터 예외 처리
            continue

        # 조회한 데이터 list에 저장
        model_list.append([year, month, day, hour, power, pred_val_1[0], pred_val_2[0]])

    model_df = pd.DataFrame(model_list, columns=["year", "month", "day", "hour", "power", "pred_val_1", "pred_val_2"])
    # 예측 오차율 계산
    for index, row in model_df.iterrows():
        year = row['year']
        month = row['month']
        day = row['day']
        hour = row['hour']
        # 실제 발전량
        real = float(row['power'])
        # 1차 예측치
        pred_1 = float(row['pred_val_1'])
        # 2차 예측치
        pred_2 = float(row['pred_val_2'])
        # 1차 예측치와 실제 발전량의 차이
        diff_1 = abs(real-pred_1)
        # 2차 예측치와 실제 발전량의 차이
        diff_2 = abs(real-pred_2)
        # 1차 예측치의 오차율 계산
        err_rate_1 = diff_1/CAPACITY*100
        # 2차 예측치의 오차율 계산
        err_rate_2 = diff_2/CAPACITY*100
        # 1차 예측치와 2차 예측치의 평균 계산
        err_rate = (err_rate_1+err_rate_2)/2
        # 오차율 누적
        sum_err_rate = sum_err_rate+err_rate
        # 시간대별 예측 오차율 메모리에 저장
        err_rate_list.append([year, month, day, hour, _model_type, err_rate])
        stack_err_rate_list.append([year, month, day, hour, _model_type, err_rate])

    # 1달 오차율 계산
    if len(err_rate_list) != 0:
        # 1달간의 오차율 평균 계산
        sum_err_rate = sum_err_rate/len(err_rate_list)
        print("MODEL: ", _model_type, ", error rate: ", sum_err_rate)
        # 1달간의 오차율 저장
        err_df = pd.DataFrame(err_rate_list, columns=["year", "month", "day", "hour", "model", "error"])
        err_df.to_csv("./"+_model_type+"_err_"+str(MONTH).zfill(2)+".csv", index=False)
# 누적 데이터 저장
if len(stack_err_rate_list) != 0:
    stack_err_df = pd.DataFrame(stack_err_rate_list, columns=["year", "month", "day", "hour", "model", "error"])
    stack_err_df.to_csv("./"+_model_type+"_stack_err_"+str(MONTH).zfill(2)+".csv", index=False)