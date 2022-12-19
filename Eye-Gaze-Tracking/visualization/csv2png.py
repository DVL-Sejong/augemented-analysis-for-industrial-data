import sys
from PIL import Image
from scipy.io import arff
import numpy as np
from argparse import ArgumentParser
import glob
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

plt.ion()
from tqdm import tqdm

def parse_args():
    # 2번 모델
    parser = ArgumentParser('Eye Movement Event parser')

    parser.add_argument('--csv-folder', type=str, required=True,
                         help='Folder which has .csv files, we use it to make png files. ex. data2/outputs/output1')
    parser.add_argument('--png-folder', type=str, required=True,
                         help='Folder to save .png files (which made from .arff files). ex. result2/result1/png')
    parser.add_argument('--img-folder', type=str, required=True,
                          help='Folder of original stimulus video files. ex. original2')

    return parser.parse_args()

# python3 arff2png.py --arff-folder 'data1/outputs/output2' --png-folder 'results1/result2/png' --img-folder 'original1'
def __main__(args):

    #import pdb;pdb.set_trace()
    evt_dict = {'UNKNOWN' : 0, 'FIX' : 1, 'SACCADE' : 2, 'SP' : 3, 'NOISE' : 4}
    data_evt_color_map = dict({
            0: 'gray',  #0. Undefined
            1: 'b',     #1. Fixation
            2: 'r',     #2. Saccade
            3: 'y',     #3. Post-saccadic oscillation
            4: 'm',     #4. Smooth pursuit
            5: 'k',     #5. Blink
            9: 'k',     #9. Other
        })
    #import pdb;pdb.set_trace()
    save = True
    
    all_filenames = sorted(glob.glob(f'{args.csv_folder}/*.csv'))
    
    for fname in tqdm(all_filenames):
        df = pd.read_csv(fname)
        
        #import pdb;pdb.set_trace()
        print(fname)
        png_spath = fname.replace(str(args.csv_folder), str(args.png_folder))
        png_spath = png_spath.replace('csv', 'png')
        
        fig = plt.figure(figsize=(10,6))
        ax00 = plt.subplot2grid((1, 2), (0, 0))
        ax01 = plt.subplot2grid((1, 2), (0, 1))

        _data = df
        ax00.plot(_data['x'], _data['y'], '-')
        ax01.plot(_data['x_pred'], _data['y_pred'], '-')

        ax00.set_xlabel('x')
        ax00.set_ylabel('y')

        ax01.set_xlabel('x')
        ax01.set_ylabel('y')

        # import pdb;pdb.set_trace()

        for e, c in data_evt_color_map.items():
            mask = _data['evt'] == e
            mask2 = _data['evt_pred'] == e
            ax00.plot(_data['x'][mask], _data['y'][mask], '.', color = c)
            ax01.plot(_data['x'][mask2], _data['y'][mask2], '.', color = c)
        #import pdb;pdb.set_trace()

        etdata_extent = np.nanmax([np.abs(_data['x']), np.abs(_data['y'])])+1

        ax00.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        #import pdb;pdb.set_trace()
        
        # image
        png_sdir = '/'.join(png_spath.split('/')[:-1])
        if not os.path.exists(png_sdir):
            os.makedirs(png_sdir)

        original_img_lst = ['data2/stimuli/images/Blad1024x768.png',
                            'data2/stimuli/images/Europe1024x768.png',
                            'data2/stimuli/images/konijntjes1024x768.png',
                            'data2/stimuli/images/Rome1024x768.png',
                            'data2/stimuli/images/vy1024x768.png']

        if 'Blad' in fname:
            original_img_path = original_img_lst[0]
        elif 'Europe' in fname:
            original_img_path = original_img_lst[1]
        elif 'konijntjes' in fname:
            original_img_path = original_img_lst[2]
        elif 'Rome' in fname:
            original_img_path = original_img_lst[3]
        elif 'vy' in fname:
            original_img_path = original_img_lst[4]
        img = plt.imread(original_img_path)

        ax00.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        ax01.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        plt.tight_layout()

        if save and not(png_spath is None):
            plt.savefig('%s' % (png_spath))
            print('>>>> saved:', png_spath)
            plt.close()
        #import pdb;pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()
    __main__(args)
