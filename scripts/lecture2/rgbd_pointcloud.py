"""RGB-D -> camera points -> world points -> CoppeliaSim coloured point cloud.
Run: python scripts/lecture2/rgbd_pointcloud.py (simulation stopped).
The cloud is displayed 0.65 m beside the scene for comparison; saved points are unshifted.
Only surfaces visible to this camera are reconstructed, not complete object meshes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from shared.common import connect, pose
from lecture2.support.bin_cell import VisionCell
from lecture2.support.utils import OUT
from lecture2.support.camera import save_ply

DISPLAY_OFFSET = [0, .65, 0]  # visual separation only; not part of camera calibration


def backproject(rgb, depth, fov, far):
    """CoppeliaSim camera: +Z forward, image-right is -X, image-down is -Y."""
    height, width = depth.shape
    f = max(width, height)/(2*np.tan(fov/2))  # focal length in pixels
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth < far*.999)
    z = depth[valid]
    # Pixel centres are u+0.5, v+0.5; images have already been flipped vertically.
    x = (width/2-(u[valid]+.5))*z/f
    y = (height/2-(v[valid]+.5))*z/f
    return np.column_stack((x, y, z)), rgb[valid]


def main():
    client, sim = connect()
    cell = VisionCell(client, sim)  # same robot-free YCB scene as vision_sensor.py
    camera = cell.h['binCamera']
    with cell.running():
        sim.step()
        rgb_bytes, (width, height) = sim.getVisionSensorImg(camera)
        depth_bytes, _ = sim.getVisionSensorDepth(camera, 1)  # metric depth [m]
        rgb = np.flipud(np.frombuffer(rgb_bytes, np.uint8).reshape(height, width, 3))
        depth = np.flipud(np.frombuffer(depth_bytes, np.float32).reshape(height, width))
        T = pose(sim, camera)  # capture the camera pose at the SAME time as RGB-D
        fov = sim.getObjectFloatParam(camera, sim.visionfloatparam_perspective_angle)
        far = sim.getObjectFloatParam(camera, sim.visionfloatparam_far_clipping)

    # 1. Depth supplies geometry; RGB supplies one colour per point.
    camera_points, colors = backproject(rgb, depth, fov, far)
    # 2. Transform from camera coordinates into world coordinates: p_w = R p_c + t.
    world_points = camera_points @ T[:3, :3].T + T[:3, 3]
    # 3. Insert world points and their RGB colours into a native point-cloud object.
    previous = sim.getObject('/RGBD_PointCloud', {'noError': True})
    if previous != -1:
        sim.removeObjects([previous])
    cloud = sim.createPointCloud(.02, 1000, 8 | 16, 3.)  # fast display + emissive colours
    sim.setObjectAlias(cloud, 'RGBD_PointCloud')
    sim.insertPointsIntoPointCloud(cloud, 2, world_points.ravel().tolist(), colors.ravel().tolist())
    sim.setObjectPosition(cloud, DISPLAY_OFFSET, sim.handle_world)
    sim.setObjectSelection([cloud])
    view = sim.getObject('/DefaultCamera', {'noError': True})
    if view != -1:
        sim.cameraFitToView(view, [cloud, *cell.h['objects'].values()], 0, 1.2)

    OUT.mkdir(parents=True, exist_ok=True)
    save_ply(str(OUT/'vision_cloud.ply'), world_points, colors/255.)
    np.savez_compressed(OUT/'vision_cloud.npz', points_world=world_points, rgb=colors,
                        depth_m=depth, T_world_camera=T)
    print(f'{len(world_points):,} coloured points. Select /RGBD_PointCloud in the scene hierarchy.')
    print('Point cloud displayed beside the original scene; saved coordinates have no display offset.')


if __name__ == '__main__':
    main()
