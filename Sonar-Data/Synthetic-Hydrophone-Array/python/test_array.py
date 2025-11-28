"""
Test script to verify the array setup
"""

import numpy as np
import matplotlib.pyplot as plt
from arrayset import (hydronum, ch_loc_map, d, Design_Freq, 
                       beam_time_delay, a, theta_select, M, c, fs)

print("="*60)
print("Array Configuration Test")
print("="*60)

print(f"\n주파수 설계 모드 (Design Frequency Mode): {Design_Freq}")
print(f"센서 간격 (Sensor spacing): {d} m")
print(f"하이드로폰 개수 (Number of hydrophones): {M}")
print(f"샘플링 주파수 (Sampling frequency): {fs} Hz")
print(f"음속 (Speed of sound): {c} m/s")

print(f"\n하이드로폰 번호 (Hydrophone numbers):")
print(f"  First 5: {hydronum[:5]}")
print(f"  Last 5: {hydronum[-5:]}")

print(f"\n채널 위치 맵 (Channel location map) shape: {ch_loc_map.shape}")
print(f"  X coordinates: min={ch_loc_map[0,:].min():.2f}, max={ch_loc_map[0,:].max():.2f}")
print(f"  Y coordinates: min={ch_loc_map[1,:].min():.2f}, max={ch_loc_map[1,:].max():.2f}")

print(f"\n빔 시간 지연 (Beam time delay) shape: {beam_time_delay.shape}")
print(f"각도 범위 (Angle range): {theta_select[0]}° to {theta_select[-1]}°")

print(f"\n가중치 벡터 (Weighting vector) length: {len(a)}")
print(f"  First 5 values: {a[:5]}")

print("\n"+"="*60)
print("Array visualization saved to 'array_geometry.png'")
print("="*60)

# Visualize array geometry
plt.figure(figsize=(12, 8))

# Plot 1: Array geometry
plt.subplot(2, 2, 1)
plt.plot(ch_loc_map[0, :], ch_loc_map[1, :], 'bo-', markersize=8)
plt.grid(True)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'Array Geometry (Design Freq Mode {Design_Freq})')
plt.axis('equal')

# Plot 2: Beam pattern angles
plt.subplot(2, 2, 2)
plt.plot(theta_select, np.ones_like(theta_select), 'r.')
plt.grid(True)
plt.xlabel('Angle (degrees)')
plt.ylabel('Beam')
plt.title(f'Beam Angles (Total: {len(theta_select)})')
plt.xlim([0, 180])

# Plot 3: Weighting vector
plt.subplot(2, 2, 3)
plt.stem(range(M), a)
plt.grid(True)
plt.xlabel('Hydrophone Index')
plt.ylabel('Weight')
plt.title('Weighting Vector')

# Plot 4: Beam time delay (first few angles)
plt.subplot(2, 2, 4)
angles_to_plot = [0, 30, 60, 90, 120, 150, 180]
for angle in angles_to_plot:
    idx = np.where(theta_select == angle)[0][0]
    plt.plot(range(M), beam_time_delay[:, idx], 'o-', label=f'{angle}°')
plt.grid(True)
plt.xlabel('Hydrophone Index')
plt.ylabel('Time Delay (s)')
plt.title('Beam Time Delay for Selected Angles')
plt.legend()

plt.tight_layout()
plt.savefig('array_geometry.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved successfully!")

print("\n배열 설정 테스트 완료! (Array setup test completed!)")