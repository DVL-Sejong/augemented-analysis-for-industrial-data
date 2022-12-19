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
    # used for deep-learning model
    parser = ArgumentParser('Eye Movement Event parser')

    parser.add_argument('--arff-folder', type=str, required=True,
                         help='Folder which has .arff files, we use it to make png files. ex. data1/outputs/output1')
    parser.add_argument('--png-folder', type=str, required=True,
                         help='Folder to save .png files (which made from .arff files). ex. result1/result1/png')
    parser.add_argument('--img-folder', type=str, required=True,
                          help='Folder of original stimulus video files. ex. original1')

    return parser.parse_args()

def __main__(args):
    
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
    
    save = True
    all_filenames = sorted(glob.glob(f'{args.arff_folder}/*/*.arff'))
    
    for fname in tqdm(all_filenames):
        data, meta = arff.loadarff(fname)
        df = pd.DataFrame(np.array(data))
        df.columns = df.columns.str.replace('time', 't')
        df.columns = df.columns.str.replace('confidence', 'status')
        df.columns = df.columns.str.replace('handlabeller_final', 'evt')
        df.columns = df.columns.str.replace('EYE_MOVEMENT_TYPE', 'evt_pred')
        
        try:
            df['evt_pred']=pd.DataFrame({ "evt_pred": [ evt_dict[re.sub(r"[^A-Z]", "", str(x))] for x in df['evt_pred'].to_list() ]})
        except Exception as e:
            print('>>>> evt_pred:', e)

        
        df['status']=pd.DataFrame({ "status": [ x == 1.0 for x in df['status'].to_list() ]})
        df['evt'] = df['evt'].astype('uint8')
        
        print(fname)
        png_spath = fname.replace(str(args.arff_folder), str(args.png_folder))
        png_spath = png_spath.replace('arff', 'png')
        
        fig = plt.figure(figsize=(10,6))
        ax00 = plt.subplot2grid((1, 2), (0, 0))
        ax01 = plt.subplot2grid((1, 2), (0, 1))

        _data = df
        ax00.plot(_data['x'], _data['y'], '-')
        ax01.plot(_data['x'], _data['y'], '-')

        ax00.set_xlabel('x')
        ax00.set_ylabel('y')

        ax01.set_xlabel('x')
        ax01.set_ylabel('y')

        for e, c in data_evt_color_map.items():
            mask = _data['evt'] == e
            mask2 = _data['evt_pred'] == e
            ax00.plot(_data['x'][mask], _data['y'][mask], '.', color = c)
            ax01.plot(_data['x'][mask2], _data['y'][mask2], '.', color = c)

        etdata_extent = np.nanmax([np.abs(_data['x']), np.abs(_data['y'])])+1

        ax00.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        png_sdir = '/'.join(png_spath.split('/')[:-1])
        if not os.path.exists(png_sdir):
            os.makedirs(png_sdir)

        img_name = png_sdir.split('/')[-1]
        original_img_path = f'{args.img_folder}/{img_name}/fps_30_000.png'
        img = plt.imread(original_img_path)
        ax00.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])
        ax01.imshow(img, extent = [-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        if save and not(png_spath is None):
            plt.savefig('%s' % (png_spath))
            print('>>>> saved:', png_spath)
            plt.close()

if __name__ == '__main__':
    args = parse_args()
    __main__(args)
