"""
Array Setting Module
배열 설정 모듈
- 하이드로폰 배열 Geometry 설정
- 빔포밍 파라미터 설정
- 주파수 설계 옵션
"""
import numpy as np
import scipy.signal as signal

# Array setting
# Note: The original code loads from .mat file - you'll need to provide x_a, y_a separately
# x_a = x_a * 1000
# y_a = y_a * 1000

fs = 2048  # Sampling frequency
st_sec = 0  # data start_time
end_sec = 60  # data end_time
Design_Freq = 1  # 1: 800Hz, 2: 400Hz, 3: 200Hz, 4: 100Hz
weighting_vector = 1  # 1: rect win, 2: hamming, 3: hanning, 4: dolph-chebyshev
NBEAM = 181
c = 1510  # Speed of sound in water (m/s)

# These would be loaded from file in original code
# main = 'D:\\데이터\\밍구해양데이터\\2. 국과연데이터'
# GPS_add = os.path.join(main, 'info')

loc_bearing = np.array([119.15, 59.7, 179.19])

# Array hydrophone Geometry
def setup_array(design_freq=1):
    """
    Setup hydrophone array geometry based on design frequency
    
    Parameters:
    -----------
    design_freq : int
        1: 800Hz, 2: 400Hz, 3: 200Hz, 4: 100Hz
    
    Returns:
    --------
    hydronum : array
        Hydrophone numbers
    ch_loc_map : ndarray (2, N)
        Channel location map [x; y] coordinates
    d : float
        Sensor spacing
    """
    
    if design_freq == 1:  # 800Hz
        hydronum = np.arange(37, 85)  # 37:84 in MATLAB (inclusive)
        d = 0.9375  # Sensor spacing [m]
        ch_loc = d * np.arange(len(hydronum))
        array_center = ch_loc[-1] / 2
        ch_loc_map = np.vstack([np.zeros(len(ch_loc)), ch_loc - array_center])
        
    elif design_freq == 2:  # 400Hz
        g1 = np.arange(25, 37)  # 25:36
        g2 = np.arange(37, 85, 2)  # 37:2:84
        g3 = np.arange(85, 97)  # 85:96
        hydronum = np.concatenate([g1, g2, g3])
        d = 1.8750
        
        ch1 = d * np.arange(len(g1))
        ch2_start = ch1[-1] + 1.4063
        ch2 = np.arange(ch2_start, ch2_start + d * len(g2), d)
        ch3_start = ch2[-1] + d/2 + 1.4063
        ch3 = np.arange(ch3_start, ch3_start + d * len(g3), d)
        
        ch_loc = np.concatenate([ch1, ch2, ch3])
        array_center = ch_loc[-1] / 2
        ch_loc_map = np.vstack([np.zeros(len(ch_loc)), ch_loc - array_center])
        
    elif design_freq == 3:  # 200Hz
        g1 = np.arange(13, 25)  # 13:24
        g2 = np.arange(25, 37, 2)  # 25:2:36
        g3 = np.arange(37, 85, 4)  # 37:4:84
        g4 = np.arange(85, 97, 2)  # 85:2:96
        g5 = np.arange(97, 109)  # 97:108
        hydronum = np.concatenate([g1, g2, g3, g4, g5])
        d = 3.75
        
        ch1 = d * np.arange(len(g1))
        ch2_start = ch1[-1] + 2.8125
        ch2 = np.arange(ch2_start, ch2_start + d * len(g2), d)
        ch3_start = ch2[-1] + d/2 + 1.4063
        ch3 = np.arange(ch3_start, ch3_start + d * len(g3), d)
        ch4_start = ch3[-1] + d/4*3 + 1.4063
        ch4 = np.arange(ch4_start, ch4_start + d * len(g4), d)
        ch5_start = ch4[-1] + d/2 + 2.8125
        ch5 = np.arange(ch5_start, ch5_start + d * len(g5), d)
        
        ch_loc = np.concatenate([ch1, ch2, ch3, ch4, ch5])
        array_center = ch_loc[-1] / 2
        ch_loc_map = np.vstack([np.zeros(len(ch_loc)), ch_loc - array_center])
        
    elif design_freq == 4:  # 100Hz
        g1 = np.arange(1, 13)  # 1:12
        g2 = np.arange(13, 25, 2)  # 13:2:24
        g3 = np.arange(25, 37, 4)  # 25:4:36
        g4 = np.arange(37, 85, 8)  # 37:8:84
        g5 = np.arange(85, 97, 4)  # 85:4:96
        g6 = np.arange(97, 109, 2)  # 97:2:108
        g7 = np.arange(109, 121)  # 109:120
        hydronum = np.concatenate([g1, g2, g3, g4, g5, g6, g7])
        d = 7.5
        
        ch1 = d * np.arange(len(g1))
        ch2_start = ch1[-1] + 5.6250
        ch2 = np.arange(ch2_start, ch2_start + d * len(g2), d)
        ch3_start = ch2[-1] + d/2 + 2.8125
        ch3 = np.arange(ch3_start, ch3_start + d * len(g3), d)
        ch4_start = ch3[-1] + d/4*3 + 1.4063
        ch4 = np.arange(ch4_start, ch4_start + d * len(g4), d)
        ch5_start = ch4[-1] + d/8*7 + 1.4063
        ch5 = np.arange(ch5_start, ch5_start + d * len(g5), d)
        ch6_start = ch5[-1] + d/4*3 + 2.8125
        ch6 = np.arange(ch6_start, ch6_start + d * len(g6), d)
        ch7_start = ch6[-1] + d/2 + 5.6250
        ch7 = np.arange(ch7_start, ch7_start + d * len(g7), d)
        
        ch_loc = np.concatenate([ch1, ch2, ch3, ch4, ch5, ch6, ch7])
        array_center = ch_loc[-1] / 2
        ch_loc_map = np.vstack([np.zeros(len(ch_loc)), ch_loc - array_center])
        
    else:
        raise ValueError("Design_Freq must be 1, 2, 3, or 4")
    
    return hydronum, ch_loc_map, d


def setup_beamforming(hydronum, ch_loc, c, NBEAM=181, weighting_vector=1):
    """
    Setup beamforming parameters
    
    Parameters:
    -----------
    hydronum : array
        Hydrophone numbers
    ch_loc : array
        Channel locations (1D array of y-coordinates)
    c : float
        Speed of sound
    NBEAM : int
        Number of beams
    weighting_vector : int
        Type of window (1: rect, 2: hamming, 3: hanning, 4: chebyshev)
    
    Returns:
    --------
    beam_time_delay : ndarray
        Beam time delays
    a : ndarray
        Weighting vector
    theta_select : ndarray
        Selected angles (0 to 180 degrees)
    """
    gap = -2 / (NBEAM - 1)
    y = np.arange(1, 1 + gap * (NBEAM - 1), gap)
    theta_select = np.arange(0, 181)  # 0~180도
    
    M = len(hydronum)
    beam_time_delay = np.outer(ch_loc, np.cos(np.deg2rad(theta_select))) / c
    
    # Weighting vector
    if weighting_vector == 1:
        a = np.ones(M)
    elif weighting_vector == 2:
        a = np.hamming(M)
    elif weighting_vector == 3:
        a = np.hanning(M)
    elif weighting_vector == 4:
        a = signal.chebwin(M, at=100)
    else:
        raise ValueError("Wrong weighting_vector option")
    
    return beam_time_delay, a, theta_select


# Main execution when imported
hydronum, ch_loc_map, d = setup_array(Design_Freq)
M = len(hydronum)
ch_loc = ch_loc_map[1, :]  # y-coordinates only
beam_time_delay, a, theta_select = setup_beamforming(hydronum, ch_loc, c, NBEAM, weighting_vector)

print("단위는 m 입니다.")
print(f"Number of hydrophones: {M}")
print(f"Design frequency mode: {Design_Freq}")