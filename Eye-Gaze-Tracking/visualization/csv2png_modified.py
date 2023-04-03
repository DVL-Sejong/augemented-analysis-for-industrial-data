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
    parser = ArgumentParser('Eye Movement Event parser')

    parser.add_argument('--csv-folder', type=str, required=True,
                         help='Folder which has .csv files, we use it to make png files. ex. data3/outputs/output_synthetic')
    parser.add_argument('--png-folder', type=str, required=True,
                         help='Folder to save .png files (which made from .arff files). ex. result3/result_synthetic/png')
    parser.add_argument('--img-folder', type=str,
                          help='Folder of original stimulus video files. ex. original3')

    return parser.parse_args()

def __main__(args):

    # fixation=0, saccades=1
    
    #evt_dict = {'UNKNOWN' : 0, 'FIX' : 1, 'SACCADE' : 2, 'SP' : 3, 'NOISE' : 4}
    data_evt_color_map = dict({
            0: 'b',     #0. Fixation
            1: 'r',     #1. Saccade
        })

    save = True
    
    all_filenames = sorted(glob.glob(f'{args.csv_folder}/*.csv'))
    
    for fname in tqdm(all_filenames):
        df = pd.read_csv(fname)

        print(fname)
        png_spath = fname.replace(str(args.csv_folder), str(args.png_folder))
        png_spath = png_spath.replace('csv', 'png')
        
        fig = plt.figure(figsize=(10,6))
        ax00 = plt.subplot2grid((1, 2), (0, 0))
        ax01 = plt.subplot2grid((1, 2), (0, 1))

        _data = df
        
        ax00.plot(_data['x_deg'], _data['y_deg'], '-')
        ax01.plot(_data['x_deg'], _data['y_deg'], '-')

        ax00.set_xlabel('x_deg')
        ax00.set_ylabel('y_deg')

        ax01.set_xlabel('x_deg')
        ax01.set_ylabel('y_deg')

        for e, c in data_evt_color_map.items():
            mask = _data['evt'] == e
            mask2 = _data['evt_pred'] == e
            ax00.plot(_data['x_deg'][mask], _data['y_deg'][mask], '.', color = c)
            ax01.plot(_data['x_deg'][mask2], _data['y_deg'][mask2], '.', color = c)

        etdata_extent = np.nanmax([np.abs(_data['x_deg']), np.abs(_data['y_deg'])])+1

        ax00.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        # image
        png_sdir = '/'.join(png_spath.split('/')[:-1])
        if not os.path.exists(png_sdir):
            os.makedirs(png_sdir)

        # img = plt.imread(original_img_path)
        # code used in colab.
        # ax00.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        # ax01.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        plt.tight_layout()

        if save and not(png_spath is None):
            plt.savefig('%s' % (png_spath))
            print('>>>> saved:', png_spath)
            plt.close()

if __name__ == '__main__':
    args = parse_args()
    __main__(args)
