"""Shared helpers for the course: connection, 4x4 math, DH table, scene builder.

Everything in the scene is created from Python, so every lesson starts from
the same, reproducible state (no Lua, no hand-made .ttt file).
Requires CoppeliaSim >= 4.8 (uses the property API) — tested target: 4.10.
"""
import math
import os
import time
from contextlib import contextmanager

import numpy as np

HOST = os.environ.get("COPPELIASIM_HOST", "localhost")
PORT = int(os.environ.get("COPPELIASIM_PORT", "23000"))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


# ---------------------------------------------------------------- connection
def connect():
    """Return (client, sim). `client.require(...)` loads other modules (simIK, simOMPL)."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print(f"connecting to CoppeliaSim at {HOST}:{PORT} ...")
    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")
    ver = sim.getInt32Param(sim.intparam_program_full_version)
    print(f"connected, CoppeliaSim version code {ver}")
    return client, sim


def stop_simulation(sim):
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.05)


@contextmanager
def stepped_simulation(sim):
    """Run the simulation in stepping mode: nothing advances until we call sim.step().

    This is the key to deterministic Python control: read state -> compute -> write
    commands -> sim.step(), exactly once per simulation time step.
    """
    stop_simulation(sim)
    # drawing objects made through the remote API survive simulation stop; clear the
    # previous run's tip traces here (not at the end, so they stay visible afterwards)
    sim.removeDrawingObject(sim.handle_all)
    sim.setStepping(True)
    sim.startSimulation()
    try:
        yield
    finally:
        stop_simulation(sim)
        sim.setStepping(False)


def first(ret):
    """Some API calls return a tuple (value, extra...). Take the first element."""
    return ret[0] if isinstance(ret, (list, tuple)) else ret


# ---------------------------------------------------------------- 4x4 math
def to_np(m12):
    """CoppeliaSim 12-value matrix (row-major 3x4) -> 4x4 numpy."""
    return np.vstack([np.asarray(m12, dtype=float).reshape(3, 4), [0, 0, 0, 1]])


def to_sim(T):
    """4x4 numpy -> CoppeliaSim 12-value list."""
    return np.asarray(T, dtype=float)[:3, :].flatten().tolist()


def trans(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


# ---------------------------------------------------------------- DH (standard / Craig-free "distal" convention)
# UR5, standard DH:  T_{i-1,i} = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
DH_TABLE = [
    # (d,        a,        alpha)
    (0.089159, 0.0, math.pi / 2),
    (0.0, -0.425, 0.0),
    (0.0, -0.39225, 0.0),
    (0.10915, 0.0, math.pi / 2),
    (0.09465, 0.0, -math.pi / 2),
    (0.0823, 0.0, 0.0),
]
Q_HOME = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
JOINT_LIMIT = math.pi  # every joint: [-pi, +pi]


def dh(theta, d, a, alpha):
    return rotz(theta) @ trans(z=d) @ trans(x=a) @ rotx(alpha)


def fk_dh(q, table=DH_TABLE):
    """Pose of frame 6 (flange) expressed in frame 0 (robot base)."""
    T = np.eye(4)
    for qi, (d, a, alpha) in zip(q, table):
        T = T @ dh(qi, d, a, alpha)
    return T


# ---------------------------------------------------------------- scene layout (world frame, metres)
TABLE_TOP = 0.25
BASE_POSE = trans(0, 0, TABLE_TOP + 0.05) @ rotz(math.pi)  # robot faces +x
TOOL = trans(z=0.14)  # flange -> grasp centre ("tip")
CUBE_SIZE = 0.05
CUBE_POS = [0.40, 0.30, TABLE_TOP + CUBE_SIZE / 2]
PLACE_POS = [0.40, -0.30, TABLE_TOP + CUBE_SIZE / 2]
WALL_SIZE = [0.40, 0.03, 0.35]
WALL_POS = [0.40, 0.0, TABLE_TOP + WALL_SIZE[2] / 2]
CAM_POS = [0.35, 0.15, 1.20]
CAM_RES = 512
CAM_FOV = math.radians(60)
CAM_NEAR, CAM_FAR = 0.05, 2.0
FINGER_OPEN, FINGER_CLOSED = 0.06, CUBE_SIZE / 2 + 0.005


def _shape(sim, kind, size, color, parent, T_rel, alias, dynamic=False, respondable=False, mass=None):
    h = sim.createPrimitiveShape(kind, size, 0)
    sim.setObjectAlias(h, alias)
    sim.setObjectParent(h, parent, True)
    sim.setObjectMatrix(h, to_sim(T_rel), sim.handle_parent)
    sim.setShapeColor(h, "", sim.colorcomponent_ambient_diffuse, color)
    # Property API (CoppeliaSim >= 4.8): every object state is a named property.
    sim.setBoolProperty(h, "dynamic", dynamic)
    sim.setBoolProperty(h, "respondable", respondable)
    for p in ("collidable", "measurable", "detectable"):
        sim.setBoolProperty(h, p, True)
    if mass is not None:
        sim.setFloatProperty(h, "mass", mass)
    return h


def _dummy(sim, alias, parent, T_rel, size=0.02, color=None):
    h = sim.createDummy(size)
    sim.setObjectAlias(h, alias)
    sim.setObjectParent(h, parent, True)
    sim.setObjectMatrix(h, to_sim(T_rel), sim.handle_parent)
    if color is not None:
        sim.setObjectColor(h, 0, sim.colorcomponent_ambient_diffuse, color)
    return h


def _cyl_between(sim, parent, p0, p1, radius, color, alias):
    """Cylinder (link visual + collision) from p0 to p1, expressed in `parent`."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        return None
    z = v / length
    x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = (p0 + p1) / 2
    return _shape(sim, sim.primitiveshape_cylinder, [radius * 2, radius * 2, length], color, parent, T, alias)


def clear_scene(sim):
    """Remove /Lab, and the KUKA of the force lessons (it stands where the Lab arm stands)."""
    stop_simulation(sim)
    root = sim.getObject("/Lab", {"noError": True})
    if root != -1:
        sim.removeObjects(sim.getObjectsInTree(root))
    for model in ("/LBRiiwa14R820", "/ForceLab", "/GravityDemo", "/HandGuiding", "/FrankaGripper", "/suctionPad"):
        # Library gripper models from earlier bin demos may be left at world level.
        h = sim.getObject(model, {"noError": True})
        if h != -1:
            sim.removeObjects(sim.getObjectsInTree(h))


def build_scene(sim, verbose=True):
    """Create table, cube, wall, camera and a UR5-like arm built *from the DH table*.

    Robot hierarchy (standard DH: joint i rotates about z_{i-1}):
        base(frame0) -> joint1 -> frame1 -> joint2 -> frame2 ... -> joint6 -> frame6 -> tip
    frame_i is placed at Tz(d_i) Tx(a_i) Rx(alpha_i) relative to joint_i, so the scene
    is *exactly* the DH model and fk_dh() must match sim.getObjectMatrix().
    """
    clear_scene(sim)
    grey, dark, blue = [0.75, 0.75, 0.78], [0.25, 0.25, 0.28], [0.20, 0.45, 0.90]

    root = sim.createDummy(0.001)
    sim.setObjectAlias(root, "Lab")

    table = _shape(sim, sim.primitiveshape_cuboid, [1.4, 1.4, 0.05], [0.55, 0.42, 0.30],
                   root, trans(0.2, 0, TABLE_TOP - 0.025), "table", respondable=True)
    cube = _shape(sim, sim.primitiveshape_cuboid, [CUBE_SIZE] * 3, [0.90, 0.10, 0.10],
                  root, trans(*CUBE_POS), "cube", dynamic=True, respondable=True, mass=0.1)
    wall = _shape(sim, sim.primitiveshape_cuboid, WALL_SIZE, [0.30, 0.70, 0.30],
                  root, trans(*WALL_POS), "wall", respondable=True)
    _dummy(sim, "placeMarker", root, trans(PLACE_POS[0], PLACE_POS[1], TABLE_TOP + 0.001),
           size=0.03, color=[0.1, 0.8, 0.1])

    # --- robot built from DH table
    base = _dummy(sim, "robot", root, BASE_POSE, size=0.05)
    _shape(sim, sim.primitiveshape_cylinder, [0.16, 0.16, 0.05], dark, base, trans(z=-0.025), "baseLink")
    joints, frames = [], [base]
    parent = base
    for i, (d, a, alpha) in enumerate(DH_TABLE, start=1):
        j = sim.createJoint(sim.joint_revolute, sim.jointmode_kinematic, 0, [0.12, 0.09])
        sim.setObjectAlias(j, f"joint{i}")
        sim.setObjectParent(j, parent, True)
        sim.setObjectMatrix(j, to_sim(np.eye(4)), sim.handle_parent)  # joint axis = z_{i-1}
        sim.setJointInterval(j, False, [-JOINT_LIMIT, 2 * JOINT_LIMIT])
        f = _dummy(sim, f"frame{i}", j, trans(z=d) @ trans(x=a) @ rotx(alpha), size=0.03)
        # link visuals: z-offset then x-offset, both rigidly attached to joint i
        _cyl_between(sim, j, [0, 0, 0], [0, 0, d], 0.045, grey, f"link{i}a")
        _cyl_between(sim, j, [0, 0, d], [a, 0, d], 0.040, grey, f"link{i}b")
        joints.append(j)
        frames.append(f)
        parent = f
    flange = frames[-1]

    # --- simple parallel gripper (static shapes; grasp = attach, see skills.py)
    _shape(sim, sim.primitiveshape_cylinder, [0.06, 0.06, 0.04], dark, flange, trans(z=0.02), "gripperMount")
    _shape(sim, sim.primitiveshape_cuboid, [0.16, 0.04, 0.02], dark, flange, trans(z=0.05), "palm")
    for side, sgn in (("L", 1), ("R", -1)):
        _shape(sim, sim.primitiveshape_cuboid, [0.01, 0.03, 0.09], blue, flange,
               trans(sgn * FINGER_OPEN, 0, 0.11), f"finger{side}")
    tip = _dummy(sim, "tip", flange, TOOL, size=0.02, color=[1, 0.5, 0])

    # IK target (free in the world, starts on the tip)
    for j, q in zip(joints, Q_HOME):
        sim.setJointPosition(j, q)
    target = _dummy(sim, "target", root, np.eye(4), size=0.03, color=[0.1, 0.9, 0.1])
    sim.setObjectMatrix(target, sim.getObjectMatrix(tip, sim.handle_world), sim.handle_world)

    # --- top-down perspective vision sensor (looks along its +z axis)
    cam = sim.createVisionSensor(
        2,  # bit1: perspective
        [CAM_RES, CAM_RES, 0, 0],
        [CAM_NEAR, CAM_FAR, CAM_FOV, 0.05, 0, 0, 0.2, 0.2, 0.2, 0, 0],
    )
    sim.setObjectAlias(cam, "camera")
    sim.setObjectParent(cam, root, True)
    sim.setObjectMatrix(cam, to_sim(trans(*CAM_POS) @ rotx(math.pi)), sim.handle_world)

    if verbose:
        print(f"scene built: {len(sim.getObjectsInTree(root))} objects under /Lab")
    return get_handles(sim)


# UR5 link masses (kg) from the UR5 URDF; radii only shape the cylinders (and so the inertias)
UR5_LINK_MASS = [3.7, 8.393, 2.275, 1.219, 1.219, 0.1879]
UR5_LINK_RADIUS = [0.060, 0.050, 0.040, 0.035, 0.035, 0.030]
# gearbox ratio ~100: rotor inertia x ratio^2 adds "armature" inertia on every joint axis (kg m^2)
UR5_ARMATURE = [0.4, 0.4, 0.2, 0.05, 0.05, 0.05]


def build_dynamic_scene(sim, q0=Q_HOME, tool=False, armature=None, verbose=True):
    """The same DH arm, but driven by the physics engine: every link is a dynamic shape with
    mass and inertia, every joint is a dynamic joint in torque (force) mode.

    A dynamic chain must alternate shape -> joint -> shape (a dummy in between breaks it), so:
        robot(static base) -> joint1 -> link1 -> joint2 -> link2 -> ... -> joint6 -> link6
    The DH frames frame0..frame6 and the tip are dummies hung on the links: they carry no mass.
    Links are not respondable (no self-collision, no contacts): pure rigid-body dynamics.
    tool=True adds a wrist force/torque sensor on the flange and a 0.2 kg tool (flange -> tip)
    behind it, so a force applied to the tool is measured, as on a real robot:
        ... -> joint6 -> link6 -> ftSensor -> tool
    armature: per-joint reflected motor inertia (MuJoCo joint property), e.g. UR5_ARMATURE.
    """
    clear_scene(sim)
    grey, dark = [0.75, 0.75, 0.78], [0.25, 0.25, 0.28]

    root = sim.createDummy(0.001)
    sim.setObjectAlias(root, "Lab")
    _shape(sim, sim.primitiveshape_cuboid, [1.4, 1.4, 0.05], [0.55, 0.42, 0.30],
           root, trans(0.2, 0, TABLE_TOP - 0.025), "table", respondable=True)

    # a primitive shape's frame is at its centre: the base cylinder's top face is DH frame 0
    base = _shape(sim, sim.primitiveshape_cylinder, [0.16, 0.16, 0.05], dark, root,
                  BASE_POSE @ trans(z=-0.025), "robot")
    T_frame = trans(z=0.025)  # pose of DH frame_{i-1} in the current parent
    frames, joints, links = [_dummy(sim, "frame0", base, T_frame, size=0.05)], [], []
    parent = base
    for i, ((d, a, alpha), m, r) in enumerate(zip(DH_TABLE, UR5_LINK_MASS, UR5_LINK_RADIUS), start=1):
        j = sim.createJoint(sim.joint_revolute, sim.jointmode_dynamic, 0, [0.12, 0.09])
        sim.setObjectAlias(j, f"joint{i}")
        sim.setObjectParent(j, parent, True)
        sim.setObjectMatrix(j, to_sim(T_frame), sim.handle_parent)  # joint axis = z_{i-1}
        sim.setJointInterval(j, True, [-math.pi, 2 * math.pi])  # cyclic: no joint limits
        sim.setIntProperty(j, "dynCtrlMode", sim.jointdynctrl_force)  # we command torques
        if armature is not None:
            sim.setFloatProperty(j, "mujoco.armature", armature[i - 1])
        # one uniform cylinder per link (each UR5 DH row has only d or only a)
        link = _cyl_between(sim, j, [0, 0, 0], [a, 0, d], r, grey, f"link{i}")
        sim.setBoolProperty(link, "dynamic", True)
        sim.setFloatProperty(link, "mass", m)  # inertia is rescaled with the mass
        T_frame = np.linalg.inv(pose(sim, link, j)) @ trans(z=d) @ trans(x=a) @ rotx(alpha)
        frames.append(_dummy(sim, f"frame{i}", link, T_frame, size=0.03))
        joints.append(j)
        links.append(link)
        parent = link
    _dummy(sim, "tip", parent, T_frame @ TOOL, size=0.02, color=[1, 0.5, 0])
    if tool:
        fs = sim.createForceSensor(0, [0, 1, 1, 0, 0], [0.05, 1, 1, 0, 0])  # no filtering, never breaks
        sim.setObjectAlias(fs, "ftSensor")
        sim.setObjectParent(fs, parent, True)
        sim.setObjectMatrix(fs, to_sim(T_frame), sim.handle_parent)  # sensor frame = flange (frame6)
        t = _cyl_between(sim, fs, [0, 0, 0], TOOL[:3, 3], 0.02, [0.20, 0.45, 0.90], "tool")
        sim.setBoolProperty(t, "dynamic", True)
        sim.setFloatProperty(t, "mass", 0.2)
    set_q(sim, joints, q0)  # initial configuration, set while the simulation is stopped

    if verbose:
        print(f"dynamic scene built: {len(sim.getObjectsInTree(root))} objects under /Lab")
    return get_dynamic_handles(sim)


def find(sim, alias):
    """Get a handle by path. '/Lab/tip' searches /Lab's whole tree for alias 'tip'
    (intermediate levels may be skipped). Falls back to an explicit tree search."""
    h = sim.getObject(f"/Lab/{alias}", {"noError": True})
    if h != -1:
        return h
    for h in sim.getObjectsInTree(sim.getObject("/Lab")):
        if sim.getObjectAlias(h) == alias:
            return h
    raise LookupError(alias)


def get_handles(sim):
    """Look objects up by *path* -- the normal way to get handles in an existing scene."""
    g = lambda a: find(sim, a)
    return {
        "root": sim.getObject("/Lab"),
        "table": g("table"),
        "cube": g("cube"),
        "wall": g("wall"),
        "robot": g("robot"),
        "joints": [g(f"joint{i}") for i in range(1, 7)],
        "frames": [g("robot")] + [g(f"frame{i}") for i in range(1, 7)],
        "tip": g("tip"),
        "fingers": [g("fingerL"), g("fingerR")],
        "target": g("target"),
        "camera": g("camera"),
        "placeMarker": g("placeMarker"),
    }


def get_dynamic_handles(sim):
    g = lambda a: find(sim, a)
    return {
        "root": sim.getObject("/Lab"),
        "robot": g("robot"),
        "joints": [g(f"joint{i}") for i in range(1, 7)],
        "links": [g(f"link{i}") for i in range(1, 7)],
        "frames": [g(f"frame{i}") for i in range(7)],
        "tip": g("tip"),
        "ftSensor": sim.getObject("/Lab/ftSensor", {"noError": True}),  # -1 without tool
        "tool": sim.getObject("/Lab/tool", {"noError": True}),
    }


def get_q(sim, joints):
    return [sim.getJointPosition(j) for j in joints]


def set_q(sim, joints, q):
    for j, qi in zip(joints, q):
        sim.setJointPosition(j, float(qi))


def pose(sim, h, rel=None):
    rel = sim.handle_world if rel is None else rel
    return to_np(sim.getObjectMatrix(h, rel))


def fmt(v, n=4):
    return np.array2string(np.asarray(v, float), precision=n, suppress_small=True)
