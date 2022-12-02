import pandas as pd
import os

csv_file = "./hit_target.csv"
df = pd.read_csv(csv_file)
is_val = df['true_validity'] == 1

flt_df = df[is_val]
# print(flt_df)
STYLE = ['2','4']
INTERVAL_STEP = ['1','2','3','4','5']
BOX_STEP = [['1', 0], ['2', 1], ['3', 2], ['4', 3]]

_list = []
for index, _r in flt_df.iterrows():
    if _r['sti'].split('/')[3].split('.')[0] == '0':
        continue

    if _r['sti'].split('/')[3].split('.')[0].split('-')[0] == '2':
        if _r['sti'].split('/')[3].split('.')[0].split('-')[2] == '0':
            continue
        else:
            _s = _r['sti'].split('/')[3].split('.')[0].split('-')[0]
            _i = _r['sti'].split('/')[3].split('.')[0].split('-')[1]
            _b = _r['sti'].split('/')[3].split('.')[0].split('-')[2]
            _target = 0
            if _b == '1':
                _target = 0
            elif _b == '2':
                _target = 1
            elif _b == '3':
                _target = 2
            else:
                _target = 3
            _t = int(_r['t'])
            _x = float(_r['avg_x'])
            _y = float(_r['avg_y'])

            _list.append([_s, _i, _target, _t, _x, _y])

gaze_df = pd.DataFrame(_list, columns=['style', 'interval', 'target', 't', 'x', 'y'])

for _s in STYLE:
    for _i in INTERVAL_STEP:
        is_style = gaze_df['style'] == _s
        is_interval = gaze_df['interval'] == _i
        _df = gaze_df[is_style & is_interval]
        outpath = "./"+_s+"-"+_i+"-0.csv"
        _df.to_csv(outpath, index=False)