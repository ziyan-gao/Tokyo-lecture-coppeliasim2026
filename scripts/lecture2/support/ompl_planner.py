"""OMPLPlanner: one reusable joint-space planning task per query, plus collision audits."""

from itertools import pairwise

import numpy as np

from shared.common import BASE_POSE, CUBE_SIZE, FINGER_CLOSED, FINGER_OPEN, TABLE_TOP, TOOL, first, fk_dh, get_q, set_q

from .gripper import attach, detach, set_fingers
from .robot import move_linear, play_joint_path, pose_from, quintic

TREE_COLORS = {0: [0.55, 0.55, 0.55], 1: [0.1, 0.5, 1.0], 2: [1.0, 0.55, 0.0]}


class OMPLPlanner:
    """Plan joint-space paths with one robot, collision scope and algorithm.

    Every query restores the visible start state and destroys its native OMPL task.
    """

    def __init__(self, client, sim, handles, collision_pairs, algorithm="RRTConnect"):
        self.client, self.sim, self.h = client, sim, handles
        self.collision_pairs, self.algorithm = collision_pairs, algorithm

    def plan(self, start, goal, *, max_time=5.0, n_states=120, tree=None, solve_attempts=1):
        """Return an (n_states, n_joints) path, or None when no exact path is found."""
        sim, joints = self.sim, self.h["joints"]
        ompl = self.client.require("simOMPL")
        task = ompl.createTask("lesson")
        n = len(joints)
        try:
            ompl.setAlgorithm(task, getattr(ompl.Algorithm, self.algorithm))
            # C-space = the 6 joints, bounds taken from the joint limits. The list flags the joints of the
            # default low-dim projection (used by KPIECE/SBL/ProjEST-type planners, not RRT/RRTConnect/PRM):
            # joints 1-3 roughly say where the arm is, the wrist mostly sets the orientation.
            # Required for every planner: without it setup() fails ("Dimension of projection needs to be > 0")
            ompl.setStateSpaceForJoints(task, joints, [int(i < 3) for i in range(n)])
            # Fraction of the state-space extent: a coarse default can skip a thin wall.
            ompl.setStateValidityCheckingResolution(task, 0.0005)
            ompl.setCollisionPairs(task, self.collision_pairs)
            ompl.setStartState(task, list(start))
            ompl.setGoalState(task, list(goal))
            ompl.setup(task)
            for label, state in (("start", start), ("goal", goal)):
                if not ompl.isStateWithinBounds(task, list(state)):
                    raise RuntimeError(f"OMPL {label} state is outside joint limits")
                if not ompl.isStateValid(task, list(state)):
                    raise RuntimeError(f"OMPL {label} state is invalid (collision)")
            solved = False
            for call in range(solve_attempts):
                ompl.solve(task, max_time)
                solved = ompl.hasExactSolution(task)
                if solved:
                    break
                if call + 1 < solve_attempts:
                    print(
                        f"{self.algorithm}: no exact path after search {call + 1}/{solve_attempts}; "
                        "continuing the same exploration tree...",
                        flush=True,
                    )
            if tree is not None:
                # vertices (joint states), tags (which tree: 1 = start, 2 = goal for bidirectional planners),
                # tag values, edges as flat (from, to) pairs, edge weights, start and goal vertex indices
                states, tags, _, edges, _, starts, goals = ompl.getPlannerData(task)
                tree.update(
                    states=np.array(states).reshape(-1, n),
                    tags=np.array(tags),
                    edges=np.array(edges, int).reshape(-1, 2),
                    starts=starts,
                    goals=goals,
                    raw_path=np.array(ompl.getPath(task)).reshape(-1, n) if solved else None,
                    solve_calls=call + 1,
                    exact_solution=bool(solved),
                )
            if not solved:
                return None
            ompl.simplifyPath(task, 1.0)
            ompl.interpolatePath(task, n_states)
            path = np.array(ompl.getPath(task)).reshape(-1, len(joints))
            return path
        finally:
            ompl.destroyTask(task)
            set_q(sim, joints, start)  # planning may leave the robot elsewhere

    def verify(self, path):
        return verify(self.sim, self.h["joints"], path, self.collision_pairs)


def make_collections(sim, h):
    """robot = everything that moves (joint1 subtree, incl. a grasped object); obstacles = table+wall."""
    robot = sim.createCollection(0)
    sim.addItemToCollection(robot, sim.handle_tree, h["joints"][0], 0)
    obstacles = sim.createCollection(0)
    for o in (h["table"], h["wall"]):
        sim.addItemToCollection(obstacles, sim.handle_single, o, 0)
    return robot, obstacles


def in_collision(sim, a, b):
    return first(sim.checkCollision(a, b)) == 1


def verify(sim, joints, path, pairs):
    """Check dense samples on every edge, restoring joints even if a collision is found.

    Sampling is a practical check, not a continuous collision-free proof.
    """
    saved = get_q(sim, joints)
    substeps = max(2, int(np.ceil(np.max(np.abs(np.diff(path, axis=0))) / 0.01)))
    dense = densify(path, substeps=substeps)
    try:
        for q in dense:
            set_q(sim, joints, q)
            if first(sim.checkCollision(*pairs)):
                raise RuntimeError("Interpolated path collides; execution cancelled.")
    finally:
        set_q(sim, joints, saved)
    return len(dense)


def tip_positions(Q):
    """World tip positions of joint configurations Q (n,6): the FK of lecture1/fk_dh.py."""
    return np.array([(BASE_POSE @ fk_dh(q) @ TOOL)[:3, 3] for q in Q])


def densify(Q, substeps=6):
    """Insert substeps between consecutive configurations (straight lines in JOINT space)."""
    s = np.linspace(0, 1, substeps + 1)[:-1, None]
    return np.vstack([a + s * (b - a) for a, b in pairwise(Q)] + [Q[-1:]])


def tree_curves(tree, substeps=6):
    """Each tree edge as the curve the tip follows: OMPL connects two states by a straight
    line in JOINT space, which is a curve in the workspace."""
    Q = tree["states"]
    return [tip_positions(densify(Q[[a, b]], substeps)) for a, b in tree["edges"]]


def _drawing(sim, kind, size, color, items, attributes=0):
    """A drawing object holding exactly `items` (flat list, 6 values per line / 3 per point).
    Both limits fail silently: maxItemCount caps the items, and one call adds a single item
    unless the handle carries sim.handleflag_addmultiple."""
    per_item = 6 if kind == sim.drawing_lines else 3
    d = sim.addDrawingObject(kind | attributes, size, 0, -1, max(1, len(items) // per_item), color)
    if items:
        sim.addDrawingObjectItem(d | sim.handleflag_addmultiple, items)  # without the flag only the 1st item is added
    return d


def draw_planner_tree(sim, tree, transparent=False):
    """Draw the explored tree (tip positions) in the scene. Returns the drawing handles."""
    handles = []
    tags = tree["tags"] if len(tree["tags"]) == len(tree["states"]) else np.zeros(len(tree["states"]), int)
    curves = tree_curves(tree)
    attributes = sim.drawing_50percenttransparency if transparent else 0
    for tag in set(tags.tolist()):
        color = TREE_COLORS.get(tag, TREE_COLORS[0])
        segs = [np.hstack([c[:-1], c[1:]]).ravel() for (a, _), c in zip(tree["edges"], curves) if tags[a] == tag]
        handles.append(
            _drawing(sim, sim.drawing_lines, 2, color, np.concatenate(segs).tolist() if segs else [], attributes)
        )
        handles.append(
            _drawing(
                sim,
                sim.drawing_points,
                8,
                color,
                tip_positions(tree["states"][tags == tag]).ravel().tolist(),
                attributes,
            )
        )
    ends = tip_positions(tree["states"][list(tree["starts"]) + list(tree["goals"])])
    handles.append(_drawing(sim, sim.drawing_spherepts, 0.03, [0.1, 0.9, 0.1], ends.ravel().tolist(), attributes))
    if tree["raw_path"] is not None:  # the path through the tree, before simplification
        c = tip_positions(densify(tree["raw_path"]))
        handles.append(_drawing(sim, sim.drawing_lines, 6, [1, 0.1, 0.1], np.hstack([c[:-1], c[1:]]).ravel().tolist()))
    return handles


def draw_tip_trace(sim, h, color=(1, 0.5, 0)):
    """Returns an on_step callback that draws the tip trajectory in the scene."""
    dr = sim.addDrawingObject(sim.drawing_lines, 3, 0, -1, 100000, list(color))
    last = [sim.getObjectPosition(h["tip"], sim.handle_world)]

    def on_step(_k):
        p = sim.getObjectPosition(h["tip"], sim.handle_world)
        sim.addDrawingObjectItem(dr, last[0] + p)
        last[0] = p

    on_step.handle = dr  # for sim.removeDrawingObject(trace.handle)
    return on_step


def plan_transport(scene, position, R):
    """One OMPL task; plan against the measured map, audit against the real geometry."""
    sim, h = scene.sim, scene.h
    goal = scene.ik.config_for(pose_from(R, position), steps=80)
    if goal is None:
        raise RuntimeError("Transport goal has no IK solution.")
    tree = {}
    planner = OMPLPlanner(scene.client, sim, h, [scene.robot, scene.world_map])
    path = planner.plan(get_q(sim, h["joints"]), goal, tree=tree, max_time=5.0, solve_attempts=6)
    if path is None:
        raise RuntimeError("RRTConnect found no exact path in the measured map.")
    plan = {
        "path_rad": path,
        "algorithm": "RRTConnect",
        "planning_tasks": 1,
        "map_checked_samples": verify(sim, h["joints"], path, [scene.robot, scene.world_map]),
        "truth_checked_samples": verify(sim, h["joints"], path, [scene.robot, scene.truth]),
        "explored_nodes": len(tree["states"]),
    }
    return plan, tree


def approach(sim, h, ik, target, R, pairs, on_step):
    q = ik.config_for(pose_from(R, target), steps=80)
    if q is None:
        raise RuntimeError("Pre-grasp pose is unreachable.")
    path = quintic(get_q(sim, h["joints"]), q, 40)
    for pair in pairs:
        verify(sim, h["joints"], path, pair)
    play_joint_path(sim, h["joints"], path, on_step=on_step)


def fingers(sim, h, close, on_step):
    a, b = (FINGER_OPEN, FINGER_CLOSED) if close else (FINGER_CLOSED, FINGER_OPEN)
    for i, value in enumerate(np.linspace(a, b, max(2, round(0.5 / sim.getSimulationTimeStep())))):
        set_fingers(sim, h, float(value))
        on_step(i)
        sim.step()


def linear(sim, h, ik, position, R, on_step):
    error = move_linear(sim, h, ik, pose_from(R, position), 1.0, on_step)
    if error > 0.003:
        raise RuntimeError(f"Local Cartesian motion tracking error: {error:.4f} m")


def grasp(sim, h, ik, position, R, on_step):
    linear(sim, h, ik, position, R, on_step)
    fingers(sim, h, True, on_step)
    attach(sim, h, h["cube"])
    linear(sim, h, ik, np.asarray(position) + [0, 0, 0.18], R, on_step)


def release(sim, h, ik, position, R, on_step):
    # Release just above the table, avoiding the map's voxel-thick support surface.
    drop = np.array([position[0], position[1], TABLE_TOP + CUBE_SIZE / 2 + 0.025])
    linear(sim, h, ik, drop, R, on_step)
    detach(sim, h, h["cube"])
    fingers(sim, h, False, on_step)
    linear(sim, h, ik, drop + [0, 0, 0.18], R, on_step)
    for i in range(round(1.0 / sim.getSimulationTimeStep())):
        on_step(i)
        sim.step()
