"""
Time Domain Data Generation for Acoustic Array Simulation
Y shape 3 distributed array simulation
- Target -> plane wave, moving
- Conventional Beamforming
- Passive cross location
- Simulated by GodMingu
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import array settings
from arrayset import (fs, c, loc_bearing, hydronum, ch_loc_map, 
                       Design_Freq, NBEAM, weighting_vector)

# Set random seed for reproducibility
np.random.seed(0)

print("Starting Y-shape distributed array simulation...")
print("Target: plane wave, moving")
print("Method: Conventional Beamforming - Passive cross location")

## Array setting
x_a = np.array([2.10230410955287, 5.60296485550706, 3.85293510521489])
y_a = np.array([4.08185831999993, 4.07222167999977, 1.04223967999952])

## Target 1 simulation setting
ST_Position = np.array([0, 1]) * 1000  # 표적 출발 위치 (m)
End_Position = np.array([8, 1]) * 1000  # 표적 도착 위치 (m)
Target_Speed = 3  # m/s -> 약 6노트

# Calculate time array
distance = np.sqrt(np.sum((End_Position - ST_Position)**2))
t = np.arange(0, distance / Target_Speed, 1/fs)

# Target heading
delta_pos = End_Position - ST_Position
Target_heading = np.rad2deg(np.arctan2(delta_pos[1], delta_pos[0]))

## Target 2 simulation setting
ST_Position2 = np.array([8, 4]) * 1000  # 표적 출발 위치 (m)
End_Position2 = np.array([5, 8]) * 1000  # 표적 도착 위치 (m)
Target_Speed2 = 2  # m/s -> 약 4노트

# Calculate time array for target 2
distance2 = np.sqrt(np.sum((End_Position2 - ST_Position2)**2))
t2 = np.arange(0, distance2 / Target_Speed2, 1/fs)

# Target 2 heading
delta_pos2 = End_Position2 - ST_Position2
Target_heading2 = np.rad2deg(np.arctan2(delta_pos2[1], delta_pos2[0]))

# Sync time arrays to minimum length
tleng = min(len(t), len(t2))
t = t[:tleng]
t2 = t2[:tleng]

# Calculate target trajectories
heading_rad = np.deg2rad(Target_heading)
Target_loc = ST_Position.reshape(2, 1) + Target_Speed * np.array([
    [np.cos(heading_rad)],
    [np.sin(heading_rad)]
]) @ t.reshape(1, -1)

heading_rad2 = np.deg2rad(Target_heading2)
Target_loc2 = ST_Position2.reshape(2, 1) + Target_Speed2 * np.array([
    [np.cos(heading_rad2)],
    [np.sin(heading_rad2)]
]) @ t2.reshape(1, -1)

# Target frequencies (Hz)
Target_f = np.array([130, 786, 257, 377, 775, 576])
k_fd = 2 * np.pi * Target_f / c  # wavenumber

Target_f2 = np.array([450, 147, 383, 260, 350, 248])
k_fd2 = 2 * np.pi * Target_f2 / c  # wavenumber

# Average target positions over time windows
time_indices = np.arange(1, int(len(t)/fs) - 5 + 1, 5)
Target_lo = np.zeros((2, len(time_indices)))
Target_lo2 = np.zeros((2, len(time_indices)))

ii2 = 0
for ii in time_indices:
    idx = np.arange((ii-1)*fs, ii*fs)
    Target_lo[:, ii2] = np.mean(Target_loc[:, idx], axis=1)
    Target_lo2[:, ii2] = np.mean(Target_loc2[:, idx], axis=1)
    ii2 += 1

## Noise level setting
p_ref = 1  # 1 µPa을 1로 설정

## Source level setting
# Target 1
SL1 = 150
p_total_rms = 10**(SL1 / 20)
num_tones = len(Target_f)
p_tone_rms = p_total_rms / np.sqrt(num_tones)
A1 = p_tone_rms * np.sqrt(2)

# Target 2
SL2 = 150
p_total_rms = 10**(SL2 / 20)
num_tones = len(Target_f2)
p_tone_rms = p_total_rms / np.sqrt(num_tones)
A2 = p_tone_rms * np.sqrt(2)

print(f"\nSimulation parameters:")
print(f"Target 1: Start {ST_Position}, End {End_Position}, Speed {Target_Speed} m/s")
print(f"Target 2: Start {ST_Position2}, End {End_Position2}, Speed {Target_Speed2} m/s")
print(f"Sampling frequency: {fs} Hz")
print(f"Speed of sound: {c} m/s")
print(f"Source levels: SL1={SL1} dB, SL2={SL2} dB")

## Main simulation loop
NL_values = [80, 85, 90, 95, 100, 105, 110]

for NL in NL_values:
    print(f"\n{'='*60}")
    print(f"Processing Noise Level: {NL} dB")
    print(f"{'='*60}")
    
    for arraynum in range(1, 4):  # 1, 2, 3
        success = f'Axis{arraynum}'
        print(f"\n  Processing {success}...")
        
        # Rotation matrix
        th = -loc_bearing[arraynum - 1]  # Convert to 0-based index
        th_rad = np.deg2rad(th)
        rotA = np.array([
            [np.cos(th_rad), -np.sin(th_rad)],
            [np.sin(th_rad), np.cos(th_rad)]
        ])
        
        # Rotate channel locations
        ch_loc_position = rotA @ np.vstack([
            np.zeros(ch_loc_map.shape[1]),
            ch_loc_map[1, :]
        ])
        
        # Translate to array position
        ch_loc_position2 = ch_loc_position + np.array([
            [x_a[arraynum - 1]],
            [y_a[arraynum - 1]]
        ])
        
        # Receiving Signal
        overlap_Beam = 0.5
        T_BF = np.zeros((len(time_indices), 181))
        current_idx2 = 0
        
        # Progress bar for time windows
        for current_idx in tqdm(time_indices, desc=f"    {success} Time windows"):
            idx = np.arange(current_idx * fs, current_idx * fs + fs)
            temp = t[idx]
            Data2 = np.zeros((fs, ch_loc_map.shape[1]))
            
            # Loop through sensors
            for Sensor_idx in range(ch_loc_map.shape[1]):
                Target1 = np.zeros(fs, dtype=complex)
                Target2 = np.zeros(fs, dtype=complex)
                
                # Calculate distances from sensor to targets
                r1 = np.sqrt(
                    (ch_loc_position2[0, Sensor_idx] - Target_lo[0, current_idx2])**2 +
                    (ch_loc_position2[1, Sensor_idx] - Target_lo[1, current_idx2])**2
                )
                r2 = np.sqrt(
                    (ch_loc_position2[0, Sensor_idx] - Target_lo2[0, current_idx2])**2 +
                    (ch_loc_position2[1, Sensor_idx] - Target_lo2[1, current_idx2])**2
                )
                
                # Generate multi-tone signals
                for freq_m in range(len(Target_f)):
                    Target1_temp = np.exp(
                        1j * (2*np.pi*Target_f[freq_m]*r1/c - 2*np.pi*Target_f[freq_m]*temp)
                    )
                    Target2_temp = np.exp(
                        1j * (2*np.pi*Target_f2[freq_m]*r2/c - 2*np.pi*Target_f2[freq_m]*temp)
                    )
                    Target1 += Target1_temp
                    Target2 += Target2_temp
                
                # Apply amplitude and distance attenuation
                Target1 = A1 * Target1 / r1
                Target2 = A2 * Target2 / r2
                
                # Add noise
                p_rms = p_ref * 10**(NL / 20)
                Noise = p_rms * np.random.randn(len(Target1))
                
                # Combine signals (real part) and noise
                Data2[:, Sensor_idx] = np.real(Target1 + Target2) + Noise
            
            # Create output directory
            output_dir = Path('./Time')
            output_dir.mkdir(exist_ok=True)
            
            # Save data
            filename = output_dir / f'Time_Domain_{success}_{(current_idx-1):04d}sec.dat'
            np.savetxt(filename, Data2, fmt='%.6e')
            
            current_idx2 += 1
        
        print(f"    {success} completed!")

print("\n" + "="*60)
print("Simulation completed successfully!")
print("="*60)