"""OMPL: pick up a cube, cross the green obstacle, then stop at the hover goal.

    docker compose run --rm lab python lecture2/ompl_pick_place.py            # RRT
    docker compose run --rm lab python lecture2/ompl_pick_place.py RRT
    python scripts/lecture2/ompl_pick_place.py RRT --wall-height .35  # original harder scene

Pipeline: camera -> pick -> IK hover goal -> OMPL -> execute -> hold and finish.

The tree the planner explored is drawn in the scene (and saved to out/ompl_tree_*.png): every node
is a joint configuration, shown at its tip position (FK); blue = tree grown from the start,
orange = from the goal (bidirectional planners), grey = single tree; red = the path through
the tree before simplification. Compare RRTConnect (a handful of nodes) with RRT or RRTstar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import os
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

from shared.common import (OUT_DIR, PLACE_POS, CUBE_SIZE, TABLE_TOP, WALL_POS, WALL_SIZE, build_scene, connect, fmt, get_q,
                    pose, set_q, stepped_simulation)
from lecture2.support.utils import save_report
from lecture2.support.robot import IK, play_joint_path, pose_from
from lecture2.support.ompl_planner import (
    TREE_COLORS, densify, draw_planner_tree, draw_tip_trace, in_collision, make_collections,
    OMPLPlanner, tip_positions, tree_curves)
from lecture2.support.camera import locate_cube
from lecture2.support.gripper import pick

ALGO = "RRT"  # selected by main; importing this module never starts a simulation
PLANNING_TIME, SOLVE_ATTEMPTS = 5., 6
GOAL_CLEARANCE = .275  # cube bottom above table; target centre Z = 0.55 m in this scene


def plot_tree(tree, path, label):
    """The explored tree in 3D, with the wall and the table top, saved to out/ompl_tree_<label>_<ALGO>.png."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    tags = tree["tags"] if len(tree["tags"]) == len(tree["states"]) else np.zeros(len(tree["states"]), int)
    for (a, _), c in zip(tree["edges"], tree_curves(tree)):
        ax.plot(*c.T, color=TREE_COLORS.get(int(tags[a]), TREE_COLORS[0]), lw=0.6, alpha=.5)
    if tree["raw_path"] is not None:
        ax.plot(*tip_positions(densify(tree["raw_path"])).T, color="r", lw=2, label="path in the tree")
    if path is not None:
        ax.plot(*tip_positions(path).T, color="k", lw=2, label="simplified path")
    x0, y0, z0 = np.array(WALL_POS) - np.array(WALL_SIZE) / 2  # wall as a translucent box
    dx, dy, dz = WALL_SIZE
    ax.bar3d(x0, y0, z0, dx, dy, dz, color=(0.3, 0.7, 0.3), alpha=0.25)
    xx, yy = np.meshgrid([-0.2, 0.8], [-0.6, 0.6])
    ax.plot_surface(xx, yy, np.full_like(xx, TABLE_TOP), color=(0.55, 0.42, 0.3), alpha=0.15)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.set_title(f"{ALGO}, {label}: {len(tree['states'])} nodes (tip positions)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"ompl_tree_{label.replace(' ', '_')}_{ALGO}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plan_and_go(q_goal, label):
    if q_goal is None:
        raise RuntimeError(f'{label}: goal IK failed')
    q_start = get_q(sim, h["joints"])
    # Fail with a precise endpoint diagnostic, and always restore the visible robot.
    try:
        for name, q in (("start", q_start), ("goal", q_goal)):
            set_q(sim, h["joints"], q)
            if in_collision(sim, robot, obstacles):
                raise RuntimeError(f"{label}: {name} configuration is in collision")
    finally:
        set_q(sim, h["joints"], q_start)

    t0 = time.time()
    tree = {}
    planner = OMPLPlanner(client, sim, h, [robot, obstacles], ALGO)
    path = planner.plan(q_start, q_goal, tree=tree,
                        max_time=PLANNING_TIME, solve_attempts=SOLVE_ATTEMPTS)
    dt = time.time() - t0
    for d in tree_drawings:  # keep only the latest tree in the scene
        sim.removeDrawingObject(d)
    tree_drawings[:] = draw_planner_tree(sim, tree, transparent=True)
    sizes = {t: int((tree["tags"] == t).sum()) for t in set(tree["tags"].tolist())}
    print(f"{label}: {ALGO} explored {len(tree['states'])} nodes {sizes} (tag: count), {len(tree['edges'])} edges")
    plot_tree(tree, path, label)
    if path is None:
        save_report('ompl_'+label.replace(' ','_'),dict(algorithm=ALGO,success=False,
            start_rad=q_start,goal_rad=q_goal,wall_size_m=WALL_SIZE,planning_s=dt,explored_nodes=len(tree['states']),
            solve_calls=tree['solve_calls'],states_rad=tree['states'],edges=tree['edges']))
        raise RuntimeError(f"{label}: {ALGO} found no exact path after {tree['solve_calls']} "
                           f"search calls (up to {PLANNING_TIME*SOLVE_ATTEMPTS:g} s). "
                           "Exploration saved; increase --retries to continue using RRT.")
    print(f"{label}: found a path in {dt:.2f} s; {len(tree['raw_path'])} tree states -> simplified,"
          f" interpolated to {len(path)}")
    # verify the interpolated path ourselves (READ collision state per waypoint)
    bad = 0
    for q in path:
        set_q(sim, h["joints"], q)
        bad += in_collision(sim, robot, obstacles)
    set_q(sim, h["joints"], q_start)
    print(f"{label}: waypoints in collision: {bad}")
    if bad:
        raise RuntimeError('Collision in returned path: execution cancelled.')
    from lecture2.support.ompl_planner import verify
    checked = verify(sim,h['joints'],path,[robot,obstacles])
    save_report('ompl_'+label.replace(' ','_'), dict(algorithm=ALGO, path_rad=path,
        checked_samples=checked, planning_s=dt, explored_nodes=len(tree['states']),
        success=True, solve_calls=tree['solve_calls'],start_rad=q_start,goal_rad=q_goal,wall_size_m=WALL_SIZE))
    play_joint_path(sim, h["joints"], path, on_step=trace)


def main():
    global client, sim, h, ik, robot, obstacles, tree_drawings, trace, ALGO, PLANNING_TIME, SOLVE_ATTEMPTS, WALL_SIZE, WALL_POS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('algorithm',nargs='?',default='RRT',choices=['RRTConnect','RRT','RRTstar','PRM'])
    parser.add_argument('--wall-height',type=float,default=.35,help='wall height in metres; .35 = original scene, .18 = optional easier scene')
    parser.add_argument('--planning-time',type=float,default=5.,help='seconds per search call')
    parser.add_argument('--retries',type=int,default=6,help='maximum search calls, retaining the same tree')
    args = parser.parse_args()
    if not np.isfinite(args.wall_height) or args.wall_height<=0:
        parser.error('--wall-height must be finite and positive')
    if not np.isfinite(args.planning_time) or args.planning_time<=0 or args.retries<1:
        parser.error('--planning-time must be finite/positive and --retries at least 1')
    ALGO, PLANNING_TIME, SOLVE_ATTEMPTS = args.algorithm,args.planning_time,args.retries
    if ALGO == 'RRT':
        print('Planner: single-tree RRT; retries retain the same task/tree. No algorithm fallback.',flush=True)
    client, sim = connect()
    h = build_scene(sim, verbose=False)
    original_size,_ = sim.getShapeBB(h['wall'])
    sim.scaleObject(h['wall'],1,1,args.wall_height/original_size[2],0)
    WALL_SIZE = [WALL_SIZE[0],WALL_SIZE[1],args.wall_height]
    WALL_POS = [WALL_POS[0],WALL_POS[1],TABLE_TOP+args.wall_height/2]
    sim.setObjectPosition(h['wall'],WALL_POS,sim.handle_world)
    print(f'Scene: wall height {args.wall_height:.2f} m | planner {ALGO} | '
          f'up to {SOLVE_ATTEMPTS} x {PLANNING_TIME:g} s per phase',flush=True)
    ik = IK(client, sim, h)
    robot, obstacles = make_collections(sim, h)


    tree_drawings = []
    hover_z = TABLE_TOP+CUBE_SIZE/2+GOAL_CLEARANCE
    final_q,final_cube,final_fingers = None,None,None
    try:
        with stepped_simulation(sim):
            for _ in range(5):
                sim.step()
            trace = draw_tip_trace(sim, h, color=(0, 0.8, 1))

            p_cube = locate_cube(sim, h)
            print("cube seen at", fmt(p_cube, 3))
            R = pose(sim, h["tip"])[:3, :3]

            pick(sim, h, ik, p_cube, R, on_step=trace)

            # One planning task, directly to the far-side hover goal; no intermediate waypoint.
            q_hover_goal = ik.config_for(pose_from(R,[PLACE_POS[0],PLACE_POS[1],hover_z]))
            plan_and_go(q_hover_goal,"transport")
            # Finish here: keep holding the cube on the other side, above the table.
            final_q = get_q(sim,h['joints'])
            final_cube = sim.getObjectMatrix(h['cube'],sim.handle_world)
            final_fingers = [sim.getObjectPosition(f,sim.handle_parent) for f in h['fingers']]
            held_position = pose(sim,h['cube'])[:3,3]
            print('Obstacle crossed. Holding cube at',fmt(held_position,3),flush=True)
            save_report('ompl_crossing',dict(algorithm=ALGO,status='hover goal reached; cube remains held',
                final_joint_rad=final_q,cube_position_world=held_position,
                cube_clearance_above_table_m=float(held_position[2]-CUBE_SIZE/2-TABLE_TOP),
                wall_height_m=args.wall_height,goal_tip_z_m=hover_z,goal_clearance_m=GOAL_CLEARANCE,planning_tasks=1))

    finally:
        # Retain the final hover configuration after CoppeliaSim resets the simulation.
        if final_q is not None:
            set_q(sim,h['joints'],final_q)
            sim.setBoolProperty(h['cube'],'dynamic',False)
            sim.setObjectParent(h['cube'],h['tip'],True)
            sim.setObjectMatrix(h['cube'],final_cube,sim.handle_world)
            for finger,position in zip(h['fingers'],final_fingers):
                sim.setObjectPosition(finger,position,sim.handle_parent)
        ik.close()
        sim.destroyCollection(robot)
        sim.destroyCollection(obstacles)


if __name__ == '__main__':
    main()
