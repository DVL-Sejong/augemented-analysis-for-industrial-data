
from argparse import ArgumentParser
import glob
from re import L

import matplotlib.pyplot as plt
from distutils.dir_util import mkpath
import cv2 as cv

plt.ion()
from tqdm import tqdm

def parse_args():
    
    parser = ArgumentParser('Eye Movement Event parser')

    parser.add_argument('--video-folder', type=str, required=True,
                         help='Folder which has video files, we use it to make png files. ex. data1/stimuli')
    parser.add_argument('--img-folder', type=str, required=True,
                         help='Folder to save first frames of videos as .png files ex. data1/stimuli_frame')
    parser.add_argument('--number', type = int, default = 10)
    parser.add_argument('--fps', type = float, default = 30.0)
    parser.add_argument('--save', type = bool, default = True)

    return parser.parse_args()

def save_frames(args, fname):
    fps = args.fps
    #save = args.save 
    num = args.number

    video_name = (fname.split('/')[-1]).split('.')[0]
    png_sdir = f'{args.img_folder}/{video_name}'
    mkpath(png_sdir)

    print(f'start saving {fname} to {png_sdir}')

    cap = cv.VideoCapture(fname)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    count, i, retaining = 0, 0, True
    #video_frames = []
    while count < frame_count and retaining:
        retaining, frame = cap.read()
        if frame is None:
            continue
        # if count in unifrom_indices :
        #     frame = cv.resize(frame, (self.size, self.size))
        #     video_frames.append(frame)
        png_spath = f'{png_sdir}/fps_{int(fps)}_{count:03d}.png'
        # video_frames.append(frame)
        cv.imwrite(png_spath, frame)
        count += 1
        if args.number == count:
            break
    cap.release()

def __main__(args):

    all_filenames = sorted(glob.glob(f'{args.video_folder}/*.mpg'))
    print(all_filenames)

    video_lst = ['beach', 'breite_strasse', 'bridge_1', 'bridge_2',
                'bumblebee', 'doves', 'ducks_boat', 'ducks_children',
                'golf', 'holsten_gate', 'koenigstrasse', 'puppies', 
                'roundabout', 'sea', 'st_petri_gate', 'st_petri_market',
                'st_petri_mcdonalds', 'street']

    # TODO
    lst = ['data1/stimuli/beach.mpg', 'data1/stimuli/breite_strasse.mpg', 'data1/stimuli/bridge_1.mpg', 'data1/stimuli/bridge_2.mpg', 
    'data1/stimuli/bumblebee.mpg', 'data1/stimuli/doves.mpg', 'data1/stimuli/ducks_boat.mpg', 'data1/stimuli/ducks_children.mpg', 
    'data1/stimuli/golf.mpg', 'data1/stimuli/holsten_gate.mpg', 'data1/stimuli/koenigstrasse.mpg', 'data1/stimuli/puppies.mpg', 
    'data1/stimuli/roundabout.mpg', 'data1/stimuli/sea.mpg', 'data1/stimuli/st_petri_gate.mpg', 'data1/stimuli/st_petri_market.mpg', 
    'data1/stimuli/st_petri_mcdonalds.mpg', 'data1/stimuli/street.mpg']

    for fname in all_filenames:
        save_frames(args, fname)
        
        
                    

if __name__ == '__main__':
    args = parse_args()
    __main__(args)
