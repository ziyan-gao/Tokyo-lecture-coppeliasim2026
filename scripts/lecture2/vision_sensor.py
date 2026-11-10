"""Capture RGB-D from the bin-picking YCB scene; no perception or grasp execution.
Run: python scripts/lecture2/vision_sensor.py
Open scripts/out/lecture2/vision_rgbd.png in the IDE (no GUI window needed).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')  # works in Docker without a desktop/display server
import matplotlib.pyplot as plt
import numpy as np
from shared.common import connect
from lecture2.support.bin_cell import VisionCell
from lecture2.support.utils import OUT


def main():
    client, sim = connect()
    cell = VisionCell(client, sim)  # bins, YCB objects and camera; no robot or gripper
    camera = cell.h['binCamera']
    with cell.running():  # settle the YCB objects
        sim.step()  # automatically render the RGB and depth images
        rgb_bytes, (width, height) = sim.getVisionSensorImg(camera)
        depth_bytes, _ = sim.getVisionSensorDepth(camera, 1)  # bit 0: metres

    # CoppeliaSim returns the bottom row first; flip both images for display.
    rgb = np.flipud(np.frombuffer(rgb_bytes, np.uint8).reshape(height, width, 3))
    depth = np.flipud(np.frombuffer(depth_bytes, np.float32).reshape(height, width))
    print(f'{sim.getObjectAlias(camera)} | RGB: {rgb.shape}, {rgb.dtype} | '
          f'Depth: {depth.shape}, {depth.dtype}, {depth.min():.3f}–{depth.max():.3f} m')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB')
    image = axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth (optical-axis Z)')
    fig.colorbar(image, ax=axes[1], label='Depth [m]')
    for axis in axes:
        axis.axis('off')
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'vision_rgbd.png', dpi=160)
    plt.imsave(OUT/'vision_rgb.png', rgb)
    plt.imsave(OUT/'vision_depth.png', depth, cmap='viridis')  # visualization only
    np.save(OUT/'vision_depth_m.npy', depth)  # original metric values, not colours
    plt.close(fig)
    print(f'Open in IDE: {OUT / "vision_rgbd.png"}')


if __name__ == '__main__':
    main()
