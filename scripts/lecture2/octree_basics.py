"""Eye-in-hand RGB-D -> static native Octree -> one RRTConnect transport -> place.
No OctoMap dependency: the map stores observed surfaces, without probabilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from lecture2.support import camera as pc
from shared.common import connect, pose, stepped_simulation, get_q, TABLE_TOP, CUBE_POS
from lecture2.support.robot import move_joints, play_joint_path
from lecture2.support.camera import save_images, read_rgbd
from lecture2.support.ompl_planner import (
    draw_tip_trace, draw_planner_tree, approach, grasp, release, plan_transport)
from lecture2.support.scenes import VIEWS, look_at, OctreeScene, OccupancyView, height_colors, PLACE_POSITION


def pick_and_place(scene):
    sim,h,ik = scene.sim,scene.h,scene.ik
    scene.insert_target()  # Freeze the map before introducing the grasp target.
    R = np.diag([1.,-1.,-1.])
    trace = draw_tip_trace(sim,h,color=(.05,.85,.95))
    pairs = [[scene.robot,scene.world_map],[scene.robot,scene.truth]]
    approach(sim,h,ik,np.array(CUBE_POS)+[0,0,.16],R,pairs,trace)
    grasp(sim,h,ik,np.array(CUBE_POS),R,trace)
    print('Target lifted; plan one transport with the held cube included.',flush=True)
    goal = PLACE_POSITION.copy(); goal[2] = .55
    plan,tree = plan_transport(scene,goal,R)
    draw_planner_tree(sim,tree,transparent=True)
    play_joint_path(sim,h['joints'],plan['path_rad'],on_step=trace)
    release(sim,h,ik,PLACE_POSITION,R,trace)
    return plan


def scan(scene):
    sim,h = scene.sim,scene.h
    display = OccupancyView(sim,h['root'],scene.resolution,TABLE_TOP,.75)
    for i,(position,target) in enumerate(VIEWS[1:]+VIEWS[:1]):
        q = scene.ik.config_for(look_at(position,target) @ np.linalg.inv(scene.tip_camera),steps=60)
        if q is None: raise RuntimeError(f'View {i}: wrist-camera pose is unreachable.')
        if not np.allclose(get_q(sim,h['joints']),q,atol=1e-5):
            move_joints(sim,h,q,1.5)
        sim.step()
        rgb,depth = scene.rgbd_camera.rgbd()
        T = pose(sim,scene.camera)
        assert np.allclose(T,pose(sim,h['tip']) @ scene.tip_camera,atol=1e-6)
        points = pc.depth_to_points(depth,scene.K,T,stride=5,max_depth=scene.far)
        points = points[np.all((points>=[.05,-.60,.20]) & (points<[.90,.60,.80]),axis=1)]
        colors = np.rint(255*height_colors(points,TABLE_TOP,.75)).astype(int)
        # Accumulate observed surfaces directly; repeated hits share a voxel.
        sim.insertVoxelsIntoOctree(scene.octree,2,points.ravel().tolist(),colors.ravel().tolist())
        centres = scene.points()
        display.refresh_points(centres)
        scene.observations.append(dict(label=f'view{i}',points=len(points),occupied=len(centres),
                                       T_world_camera=T.tolist()))
        print(f'View {i+1}/5: {len(points)} points -> {len(centres)} static voxels',flush=True)
        save_images(rgb,depth,f'octree_view{i}')
    scene.save_map()  # Map stays unchanged during all subsequent robot motions.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voxel',type=float,default=.02)
    args = parser.parse_args()
    if not .005<=args.voxel<=.1: parser.error('Voxel size must be between .005 and .1 m.')
    client,sim = connect()
    with OctreeScene(client,sim,args.voxel) as scene:
        with stepped_simulation(sim):
            scan(scene)
            plan = pick_and_place(scene)
            scene.finish(plan)


if __name__=='__main__':
    main()
