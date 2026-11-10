"""Learned bin picking: RGB-D -> GR-ConvNet -> clearance checks -> physical grasp.
python scripts/lecture2/bin_picking_grconvnet.py --detect
python scripts/lecture2/bin_picking_grconvnet.py               # continue until empty
python scripts/lecture2/bin_picking_grconvnet.py --attempts 5  # optional attempt cap
Defaults: Bullet 2.83, friction 1.5, live RGB-D point cloud, 3 YCB objects, native RG2 gripper.
Flange -> F/T sensor -> RG2 (identical mounting poses). Physical contact/friction only.
Scene and execution helpers live in support/bin_cell.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from contextlib import nullcontext
import math
import os

import cv2
import numpy as np
import torch
from lecture2.support import grconvnet
from lecture2.support import camera as pc
from shared.common import connect, OUT_DIR
from lecture2.support.bin_cell import BIN, FLOOR, CAM_RES, BinCell
from lecture2.support.utils import save_report


def run_picking(cell, model, live, out, report, detect, draw, max_attempts=0, detect_only=False):
    attempts = empty_frames = idle = cycle = 0
    failed = []
    while True:
        cycle += 1
        # Retry with denser network peaks and a depth-based narrow-axis refinement.
        dense = idle > 0
        rgb, depth, quality, grasps = detect(cell,model,dense=dense)
        pixels = cell.visible_object_pixels  # actual unobstructed frame, after parking
        grasps = [g for g in grasps if all(np.linalg.norm(g['top'][:2]-p)>.025 for p in failed)]
        if live is not None:
            live.predictions(cell.network_predictions,grasps)
        np.savez_compressed(out/f'capture_{cycle:02d}.npz',rgb=rgb,depth_m=depth,quality=quality)
        draw(rgb,depth,quality,grasps,str(out/f'attempt_{cycle:02d}.png'))
        entry = dict(cycle=cycle,clear_candidates=len(grasps),object_pixels=pixels,recovery=dense)
        report['attempts'].append(entry)
        if detect_only:
            entry['status']=report['status']='detection only'
            return
        if pixels < 20:
            empty_frames += 1
            entry['status']='confirming empty bin'
            if empty_frames >= 3:
                report['status']='source bin empty'
                print('Source bin empty: confirmed in three RGB-D frames.',flush=True)
                return
            cell.wait(.2)
            continue
        empty_frames = 0
        if max_attempts and attempts >= max_attempts:
            report['status']='attempt limit reached'
            return
        chosen, plan = next(((g,p) for g in grasps if (p:=cell.plan(g)) is not None),(None,None))
        if plan is None:
            idle += 1
            failed = []  # exclusions expire: do not blacklist the last object forever
            entry['status']='retrying detection'
            if idle >= 5:
                report['status']='stalled: objects visible but no feasible grasp'
                print('Objects remain visible; five recovery scans found no feasible grasp.',flush=True)
                return
            print(f'Rescan {idle}/5: {pixels} object pixels; enabling dense/depth-refined proposals.',flush=True)
            cell.wait(.5)
            continue
        attempts += 1
        if live is not None:
            live.predictions(cell.network_predictions,grasps,chosen)
        ok, why = cell.execute(plan)
        if live is not None:
            live.predictions([],[])
        entry.update(sensor_success=bool(ok),reason=why,ground_truth_after=cell.where(),
                     grasp={k:v for k,v in chosen.items() if k!='ends'})
        print(f'Attempt {attempts}: {ok}: {why}',flush=True)
        if why.startswith(('INVALID_DYNAMICS','RELEASE_FAILED')):
            report['status']=why
            return
        idle = 0 if ok else idle+1
        failed = [] if ok else failed+[chosen['top'][:2]]
        if idle >= 5:
            report['status']='stalled: five consecutive grasp failures'
            return


def clear_grasp(cell, g, cloud):
    """Check open fingers AND palm footprint, not just the predicted centre pixel."""
    top, jaw = g['top'], g['jaw']
    if not np.isfinite(top).all() or top[2] < FLOOR+.012:
        return False  # no floor/rim grasp
    if not .008 <= g['opening'] <= cell.max_opening-.012:
        return False
    across = np.array([-jaw[1],jaw[0]])
    gap = min(g['opening']+.02,cell.max_opening)
    low = max(top[2]-.03,FLOOR+.025)-.02
    for sign in (-1,1):
        centre = top[:2]+sign*(gap/2+.0105)*jaw
        corners = np.array([centre+a*.0105*jaw+b*.012*across for a in (-1,1) for b in (-1,1)])
        if np.any(np.abs(corners-BIN) >= cell.bin_inner/2-.012):
            return False
        d = cloud[:,:2]-centre
        inside = (np.abs(d@jaw)<.0135)&(np.abs(d@across)<.015)&(cloud[:,2]>low)&(cloud[:,2]<top[2]+.06)
        if np.count_nonzero(inside)>3:
            return False
    palm = np.array([top[:2]+a*(cell.max_opening+.048)/2*jaw+b*.0275*across
                     for a in (-1,1) for b in (-1,1)])
    return bool(np.all(np.abs(palm-BIN)<cell.bin_inner/2-.012))


def detect(cell, model, dense=False):
    """Park outside the camera view, then infer from the current RGB-D frame."""
    rgb, depth, T = cell.capture()
    q, grasps = predict_frame(cell,model,rgb,depth,T,dense)
    return rgb,depth,q,grasps


def predict_frame(cell, model, rgb, depth, T, dense=False):
    """Learned proposals + measured-depth clearance; no object pose/CAD inputs."""
    K = cell.camera.intrinsics()
    cloud = pc.depth_to_points(depth,K,T,max_depth=1.2)
    q, angle, width = model.predict(depth)
    pixels = pc.pixel_points(depth,K,T)
    object_pixels = (pixels[:,:,2]>FLOOR+.008)&(pixels[:,:,2]<FLOOR+.20)&np.all(
        np.abs(pixels[:,:,:2]-BIN)<cell.bin_inner/2-.012,axis=2)
    cell.visible_object_pixels = int(object_pixels.sum())
    regions = pc.depth_segments(depth,object_pixels)
    quality_in_bin = np.where(object_pixels,q,0.)  # rim peaks must not crowd out object peaks
    grasps = []
    cell.network_predictions = []
    for r,c,a,w,quality in grconvnet.best_grasps(quality_in_bin,angle,width,
            n=120 if dense else 40,min_distance=4 if dense else 10,threshold=.1 if dense else .2):
        if not np.isfinite(w) or w<=0:
            continue
        direction = np.array([math.cos(a),-math.sin(a)])
        ends = [np.array([c,r])+s*direction*w/2 for s in (-1,1)]
        def world(uv):
            u,v=uv;z=depth[r,c];f,cx,cy=K[:3]
            return T[:3,:3]@np.array([(cx-u-.5)*z/f,(cy-v-.5)*z/f,z])+T[:3,3]
        p0,p1 = map(world,ends)
        delta = (p1-p0)[:2]
        opening = np.linalg.norm(delta)
        if opening<1e-6:
            continue
        top, jaw = world((c,r)), delta/opening
        cell.network_predictions.append(dict(top=top,jaw=jaw,network_opening=float(opening),
                                             quality=float(quality)))
        region = next((m for m in regions if m[r,c]),None)
        if region is None:
            continue
        # Width is refined from visible geometry; recovery can also try the region's narrow axis.
        axes = [('network',jaw)]
        if dense:
            _, basis, _ = pc.principal_axes(pixels[region,:2])
            axes.append(('depth_region_minor_axis',basis[:,-1]))
        for source, axis in axes:
            centre = top.copy()
            along = (pixels[region,:2]-centre[:2])@axis
            if dense:
                centre[:2] += np.clip((along.min()+along.max())/2,-.02,.02)*axis
                along = (pixels[region,:2]-centre[:2])@axis
            measured_width = 2*np.max(np.abs(along))+.004
            g = dict(top=centre,jaw=axis,opening=float(measured_width),network_opening=float(opening),
                     quality=float(quality),pixel=(r,c),orientation_source=source,
                     ends=pc.project(np.array([centre+s*np.r_[axis,0]*measured_width/2 for s in (-1,1)]),K,T))
            if clear_grasp(cell,g,cloud):
                grasps.append(g)
    return q,grasps


def draw(rgb, depth, q, grasps, path):
    """RGB | depth | quality map, the grasps drawn as jaw lines (best: red), 3x enlarged."""
    def big(img):
        return cv2.resize(img, (CAM_RES * 3, CAM_RES * 3), interpolation=cv2.INTER_NEAREST)
    d = cv2.applyColorMap((255 * (depth - depth.min()) / max(1e-6, np.ptp(depth))).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    qm = cv2.applyColorMap((255 * np.clip(q, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    panels = [big(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), big(d), big(qm)]
    for k, g in enumerate(reversed(grasps)):
        best = k == len(grasps) - 1
        p0, p1 = (tuple(int(3 * v) for v in e) for e in g["ends"])
        for img in panels:
            cv2.line(img, p0, p1, (40, 40, 255) if best else (255, 255, 255), 3 if best else 1, cv2.LINE_AA)
            cv2.circle(img, (int(3 * g["pixel"][1]), int(3 * g["pixel"][0])), 4, (40, 40, 255) if best else (255, 255, 255), -1)
    for img, title in zip(panels, ("camera", "depth", "grasp quality")):
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(path, np.hstack(panels))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detect',action='store_true')
    parser.add_argument('--newheap',action='store_true')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--attempts',type=int,default=0,help='grasp attempt limit; 0 = continue until empty')
    parser.add_argument('--engine',choices=('bullet','mujoco'),default='bullet')
    parser.add_argument('--bullet-version',type=int,choices=(278,283),default=283,help='Bullet 2.78 or 2.83')
    parser.add_argument('--friction',type=float,default=1.5,help='Bullet friction parameter on pads, objects and bin')
    parser.add_argument('--mass-scale',type=float,default=.5,help='YCB object mass multiplier; default 0.5')
    parser.add_argument('--no-pointcloud',action='store_true',help='disable live point-cloud display')
    args=parser.parse_args()
    if args.attempts<0: parser.error('--attempts must be nonnegative')
    if not np.isfinite(args.friction) or args.friction<0: parser.error('--friction must be finite and nonnegative')
    if not np.isfinite(args.mass_scale) or args.mass_scale<=0: parser.error('--mass-scale must be finite and positive')
    torch.set_num_threads(1)  # batch-one CPU inference; avoid thread overhead in the live loop
    model=grconvnet.GraspModel()
    client,sim=connect()
    engine=sim.physics_bullet if args.engine=='bullet' else sim.physics_mujoco
    cell=BinCell(client,sim,seed=args.seed,new_heap=args.newheap,engine=engine,friction=args.friction,mass_scale=args.mass_scale,bullet_version=args.bullet_version)
    from lecture2.support.camera import LivePointCloud
    live = None if args.no_pointcloud else LivePointCloud(cell)
    if live is not None:
        live.predictor = lambda rgb,depth,T: predict_frame(cell,model,rgb,depth,T)
    print(f'Engine: {args.engine}; object mass scale: {args.mass_scale:g}; live point cloud: {live is not None}',flush=True)
    out=Path(OUT_DIR)/'bin_picking_grconvnet';out.mkdir(parents=True,exist_ok=True)
    report=dict(gripper='RG2',mount='flange -> RG2_mount_FT -> RG2; identical sensor/gripper pose',objects=cell.object_names,bin_inner_m=cell.bin_inner,max_opening_m=cell.max_opening,
                seed=args.seed,engine=args.engine,bullet_version=args.bullet_version if args.engine=='bullet' else None,bullet_friction=args.friction,mass_scale=args.mass_scale,object_masses_kg=cell.object_masses,attempts=[],note='Physical friction grasp; no attach or forced detach. '
                'Only local gripper clearance is checked; not full-arm collision-free planning.')
    final_poses = {}
    try:
        with (live if live is not None else nullcontext()), cell.running():
            report['engine_readback']=cell.actual_engine
            report['initial_ground_truth']=cell.where()  # evaluation only
            try:
                run_picking(cell,model,live,out,report,detect,draw,args.attempts,args.detect)
            finally:
                report['final_ground_truth']=cell.where()
                report['delivered_objects']=len(report['final_ground_truth'].get('placeBin',[]))
                final_poses = {body:sim.getObjectMatrix(body,sim.handle_world) for body in cell.h['objects'].values()}
    except KeyboardInterrupt:
        report['status']='interrupted'
    finally:
        # CoppeliaSim resets dynamic objects when stopping: keep the final demo layout visible.
        for body, matrix in final_poses.items():
            sim.setObjectMatrix(body,matrix,sim.handle_world)
        if live is not None:
            report['pointcloud']=dict(updates=live.updates,prediction_updates=live.prediction_updates,
                                     final_points=live.count,period_sim_s=live.period)
        save_report(f'bin_picking_grconvnet_seed{args.seed}',report)
        save_report('bin_picking_grconvnet',report)
    print(f"Objects in destination bin: {report.get('delivered_objects',0)}; images: {out}")


if __name__=='__main__':
    main()
