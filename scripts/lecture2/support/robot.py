"""Robot handles, Cartesian position control, IK and trajectory execution."""

import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from shared.common import clear_scene, set_q, to_sim

MODEL = "/models/robots/non-mobile/KUKA LBR iiwa 14 R820.ttm"
PATH = "/LBRiiwa14R820"
HOME = np.radians([0, 30, 0, -60, 0, 30, 0])  # as in lecture1/jacobian_demo.py: tool pointing down


class Robot:
    """Bound KUKA model: reuse handles instead of passing them to every API call."""

    def __init__(self, sim, q=HOME, ctrl=None):
        self.sim = sim
        self.h = load(sim, q, ctrl)
        self.joints, self.eef = self.h["joints"], self.h["eef"]

    @classmethod
    def from_handles(cls, sim, handles):
        """Reuse an existing model without loading it or resetting any joint."""
        instance = cls.__new__(cls)
        instance.sim, instance.h = sim, handles
        instance.joints, instance.eef = handles["joints"], handles["eef"]
        return instance

    def joint_positions(self):
        return get_q(self.sim, self.joints)

    def set_targets(self, positions):
        if len(positions) != len(self.joints):
            raise ValueError("Provide one target per joint.")
        for joint, value in zip(self.joints, positions):
            self.sim.setJointTargetPosition(joint, float(value))

    def pose(self):
        return matrix(self.sim, self.eef)

    def jacobian(self):
        return geometric_jacobian(self.sim, self.joints, self.pose()[:3, 3])

    def controller(self, dt):
        return CartesianController(self.sim, self.h, dt)


def load(sim, q=HOME, ctrl=None):
    """Reuse the KUKA in the scene (or load it), switch its demo script off, put it at q.

    ctrl: joint dynamic control mode, default position control (sim.jointdynctrl_position).
    Returns a dict of handles: robot, joints (7), eef (the flange 'connection' point), flange (the
    dynamic shape the connection is attached to: a tool or sensor goes on it)."""
    robot = sim.getObject(PATH, {"noError": True})
    if robot == -1:
        robot = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir) + MODEL)
    path = sim.getObjectAlias(robot, 1)
    joints = [sim.getObject(path + "/joint", {"index": i}) for i in range(7)]
    eef = sim.getObject(path + "/connection")
    script = sim.getScript(sim.scripttype_simulation, robot)
    if script != -1:
        sim.setScriptInt32Param(script, sim.scriptintparam_enabled, 0)  # we are the controller
    ctrl = sim.jointdynctrl_position if ctrl is None else ctrl
    for j, qi in zip(joints, q):
        sim.setJointMode(j, sim.jointmode_dynamic, 0)
        sim.setIntProperty(j, "dynCtrlMode", ctrl)
        sim.setJointPosition(j, float(qi))
        sim.setJointTargetPosition(j, float(qi))
    flange = sim.getObjectParent(eef)
    while not sim.getBoolProperty(flange, "dynamic"):
        flange = sim.getObjectParent(flange)
    return {"robot": robot, "joints": joints, "eef": eef, "flange": flange}


def remove(sim):
    robot = sim.getObject(PATH, {"noError": True})
    if robot != -1:
        sim.removeModel(robot)


def matrix(sim, h):
    """World pose of an object as a 4x4 numpy array."""
    return np.vstack([np.array(sim.getObjectMatrix(h, sim.handle_world)).reshape(3, 4), [0, 0, 0, 1]])


def get_q(sim, joints):
    return np.array([sim.getJointPosition(j) for j in joints])


def geometric_jacobian(sim, joints, point):
    """6 x n world-frame Jacobian of a point on the last link (lecture1/jacobian_demo.py):
    column i = [z_i x (p - o_i); z_i] for a revolute joint i."""
    p = np.asarray(point, float)
    J = np.zeros((6, len(joints)))
    for i, j in enumerate(joints):
        M = matrix(sim, j)
        J[:3, i] = np.cross(M[:3, 2], p - M[:3, 3])
        J[3:, i] = M[:3, 2]
    return J


def dls(J, v, damping=0.01):
    """Damped least squares: the joint velocity that best produces the twist v."""
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(J)), v)


def orientation_error(R, R_goal):
    """Small-angle rotation vector that turns R into R_goal (world frame)."""
    return 0.5 * sum(np.cross(R[:, i], R_goal[:, i]) for i in range(3))


class CartesianController:
    """Move the flange along a Cartesian reference with the joints in POSITION control.

    It keeps a reference configuration q_ref (the joint targets) and, every step,
        e  = x_ref - x(q_ref),   x(q_ref) ~ x + Jv (q_ref - q)     (where the TARGETS would put the flange)
        q_ref += dt * J^+ [v_ref + kp e; kr e_R]  ->  sim.setJointTargetPosition
    and the joint servos (inside the physics engine) do the rest. The error is taken at q_ref, not at
    the measured q: when a contact blocks the arm, q_ref does not keep winding up (and then push
    harder and harder). force_sensor.py uses it for the guarded move."""

    def __init__(self, sim, k, dt, kp=20.0, kr=10.0):
        self.sim, self.k, self.dt, self.kp, self.kr = sim, k, dt, kp, kr
        self.q_ref = get_q(sim, k["joints"])
        T = matrix(sim, k["eef"])
        self.x_ref, self.R_ref = T[:3, 3].copy(), T[:3, :3].copy()

    def hold(self):
        """Forget the reference: hold where the arm really is now (e.g. after a contact)."""
        self.q_ref = get_q(self.sim, self.k["joints"])
        self.x_ref = self.position()

    def position(self):
        return np.array(self.sim.getObjectPosition(self.k["eef"], self.sim.handle_world))

    def step(self, x_ref=None, v_ref=None):
        """Send one set of joint targets (call once per sim.step()). Returns the flange position."""
        if x_ref is not None:
            self.x_ref = np.asarray(x_ref, float)
        v_ref = np.zeros(3) if v_ref is None else np.asarray(v_ref, float)
        T = matrix(self.sim, self.k["eef"])
        J = geometric_jacobian(self.sim, self.k["joints"], T[:3, 3])
        dq = self.q_ref - get_q(self.sim, self.k["joints"])  # servo lag (or a contact)
        e_x = self.x_ref - (T[:3, 3] + J[:3] @ dq)
        e_R = orientation_error(T[:3, :3], self.R_ref) - J[3:] @ dq
        self.q_ref = self.q_ref + self.dt * dls(J, np.r_[v_ref + self.kp * e_x, self.kr * e_R])
        for j, qi in zip(self.k["joints"], self.q_ref):
            self.sim.setJointTargetPosition(j, float(qi))
        return T[:3, 3]

    def move_to(self, x_goal, duration):
        """Straight line to x_goal with a smooth (quintic) time law; steps the simulation."""
        x0, n = self.x_ref.copy(), max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            s = i / n
            self.step(x0 + (10 * s**3 - 15 * s**4 + 6 * s**5) * (np.asarray(x_goal) - x0))
            self.sim.step()


def arrow_segments(origin, vector):
    """Three line segments: shaft plus arrowhead; no arrow for a zero vector."""
    origin, vector = np.asarray(origin, float), np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if length < 1e-5:
        return []
    direction = vector / length
    side = np.cross(direction, [0, 0, 1] if abs(direction[2]) < 0.9 else [0, 1, 0])
    side /= np.linalg.norm(side)
    tip = origin + vector
    size = min(0.04, length * 0.25)
    return [np.r_[origin, tip]] + [np.r_[tip, tip - size * direction + s * 0.5 * size * side] for s in (-1, 1)]


class IK:
    """Damped-least-squares IK on the chain robot-base -> tip, tracking the 'target' dummy."""

    def __init__(self, client, sim, h):
        self.sim, self.h = sim, h
        self.simIK = simIK = client.require("simIK")
        self.env = simIK.createEnvironment()
        self.group = simIK.createGroup(self.env)
        simIK.setGroupCalculation(self.env, self.group, simIK.method_damped_least_squares, 0.05, 50)
        _, self.sim2ik, _ = simIK.addElementFromScene(
            self.env, self.group, h["robot"], h["tip"], h["target"], simIK.constraint_pose
        )

    def solve(self):
        """Read target from the scene, solve, write joints back to the scene."""
        res, _, precision = self.simIK.handleGroup(self.env, self.group, {"syncWorlds": True})
        return res == self.simIK.result_success, precision

    def config_for(self, T_goal, steps=30, seed=None):
        """Joint values that put the tip at T_goal, computed *inside the IK world only*
        (the scene robot does not move). Walks the target there in small steps,
        which keeps damped least squares well behaved. seed: start from these joint values
        instead of the robot's current ones (a nearby start converges far more reliably)."""
        sim, simIK = self.sim, self.simIK
        simIK.syncFromSim(self.env, [self.group])
        if seed is not None:
            for j, qj in zip(self.h["joints"], seed):
                simIK.setJointPosition(self.env, self.sim2ik[j], float(qj))
        ik_target = self.sim2ik[self.h["target"]]
        ik_tip = self.sim2ik[self.h["tip"]]
        T0 = simIK.getObjectMatrix(self.env, ik_tip)
        ok = False
        for k in range(1, steps + 1):
            M = sim.interpolateMatrices(T0, to_sim(T_goal), k / steps)
            simIK.setObjectMatrix(self.env, ik_target, M)
            ok = simIK.handleGroup(self.env, self.group)[0] == simIK.result_success
        q = [simIK.getJointPosition(self.env, self.sim2ik[j]) for j in self.h["joints"]]
        return q if ok else None

    def close(self):
        self.simIK.eraseEnvironment(self.env)


def quintic(q0, q1, n):
    """Joint-space trajectory with zero start/end velocity and acceleration.
    s(t) = 10t^3 - 15t^4 + 6t^5, t in [0,1]."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    t = np.linspace(0, 1, n)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return q0 + np.outer(s, q1 - q0)


def play_joint_path(sim, joints, path, on_step=None):
    """Write one configuration per simulation step (kinematic joints)."""
    for k, q in enumerate(path):
        set_q(sim, joints, q)
        if on_step:
            on_step(k)
        sim.step()


def move_joints(sim, h, q_goal, duration=2.0, on_step=None):
    dt = sim.getSimulationTimeStep()
    n = max(2, int(duration / dt))
    play_joint_path(sim, h["joints"], quintic(get_q(sim, h["joints"]), q_goal, n), on_step)


def move_linear(sim, h, ik, T_goal, duration=1.5, on_step=None):
    """Cartesian straight line: interpolate the target pose, solve IK every step."""
    dt = sim.getSimulationTimeStep()
    n = max(2, int(duration / dt))
    T0 = sim.getObjectMatrix(h["tip"], sim.handle_world)
    sim.setObjectMatrix(h["target"], T0, sim.handle_world)
    worst = 0.0
    for k in range(1, n + 1):
        t = k / n
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        sim.setObjectMatrix(h["target"], sim.interpolateMatrices(T0, to_sim(T_goal), s), sim.handle_world)
        _, precision = ik.solve()
        worst = max(worst, precision[0])
        if on_step:
            on_step(k)
        sim.step()
    return worst


def pose_from(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class GravityPanel:
    def __init__(self, force_scale):
        self.force_scale = force_scale
        self.client = RemoteAPIClient(
            host=os.getenv("COPPELIASIM_HOST", "localhost"), port=int(os.getenv("COPPELIASIM_PORT", "23000"))
        )
        self.sim, self.api = self.client.require("sim"), self.client.require("simUI")
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("Stop simulation before opening this demo")
        clear_scene(sim)  # the Lab arm of the other lessons stands where the KUKA stands
        kuka = Robot(sim, HOME).h
        self.robot, self.joints, self.eef = kuka["robot"], kuka["joints"], kuka["eef"]
        old = sim.getObject("/GravityDemo", {"noError": True})
        if old != -1:
            sim.removeObjects(sim.getObjectsInTree(old))
        self.root = sim.createDummy(0.001)
        sim.setObjectAlias(self.root, "GravityDemo")
        self.materials = []
        for h in sim.getObjectsInTree(self.robot, sim.sceneobject_shape, 0):
            ok, opacity = sim.getShapeColor(h, "@compound", sim.colorcomponent_transparency)
            if ok:
                self.materials.append((h, opacity))
                sim.setShapeColor(h, "@compound", sim.colorcomponent_transparency, [min(x, 0.45) for x in opacity])
        self._create_panel()
        flags = sim.drawing_local | sim.drawing_painttag | sim.drawing_overlay
        self.coms = sim.addDrawingObject(sim.drawing_spherepoints | flags, 0.015, 0, self.root, 8, [1, 0.8, 0])
        self.weights = sim.addDrawingObject(sim.drawing_lines | flags, 4, 0, self.root, 24, [1, 0.1, 0.1])
        self.radius = sim.addDrawingObject(sim.drawing_lines | flags, 3, 0, self.root, 3, [1, 0.45, 0])
        self.joint_marker = sim.addDrawingObject(sim.drawing_spherepoints | flags, 0.022, 0, self.root, 1, [1, 1, 0])
        self.axis = sim.addDrawingObject(sim.drawing_lines | flags, 3, 0, self.root, 3, [0.8, 0, 1])
        self._last_q = None
        self._last_payload = None
        self._names = []
        self.tests = {}
        for i, q in enumerate(np.degrees(HOME), 1):
            self.api.setSliderValue(self.ui, i, int(q), True)

    def _create_panel(self):
        sliders = "".join(
            f'<label text="Joint {i} (deg)"/><hslider id="{i}" minimum="-120" maximum="120"/>' for i in range(1, 8)
        )
        xml = f"""<ui title="Gravity compensation | COM Jacobians" closeable="false" layout="vbox">
<label text="tau_load = sum(Jv_COM^T * m * gravity); tau_hold = -tau_load"/>
<label text="Robot link masses and COMs are read from the model; inertia is not needed for static gravity."/>
<group layout="form">{sliders}
<label text="Point payload at EEF (0.1 kg units)"/><hslider id="8" minimum="0" maximum="50"/>
<label text="Select joint for its moment arm"/><combobox id="9">{"".join(f"<item>Joint {i}</item>" for i in range(1, 8))}</combobox>
<label text="Select body COM"/><combobox id="10"><item>Link 1</item></combobox></group>
<label text="Yellow: COMs | Red: gravity force | Orange: joint-to-COM | Purple: selected joint axis"/>
<label text="Force arrow scale: 0.008 m/N. Torque values use signed model joint axes."/>
<label id="20" text=""/><label id="21" text=""/>
<group layout="hbox"><button text="Upright" on-click="upright"/><button text="Bent / extended" on-click="bent"/></group>
<group layout="hbox"><button text="Release: 1 s without compensation" on-click="release"/>
<button text="Hold: 1 s Jacobian compensation" on-click="compensate"/></group>
<label text="Physics test: Bullet torque mode; identical small damping, no position feedback."/>
<label text="The external point payload adds gravity only; it adds no simulated inertia."/>
<label id="22" text="Both tests reset to the same slider posture; compare joint drift."/>
<button text="Exit" on-click="exitDemo"/></ui>"""
        code = """local api=require('simUI')
local action=''
function sysCall_init() ui=api.create([=[@INSERT@]=]) end
function getPanel() return ui end
function upright() action='upright' end
function bent() action='bent' end
function release() action='release' end
function compensate() action='compensate' end
function exitDemo() action='exit' end
function takeAction() local a=action;action='';return a end
function sysCall_cleanup() if ui then api.destroy(ui) end end
""".replace("@INSERT@", xml)
        script = self.sim.createScript(self.sim.scripttype_customization, code, 0, "lua")
        self.sim.setObjectParent(script, self.root, True)
        self.sim.initScript(script)
        self.script = script
        self.ui = self.sim.callScriptFunction("getPanel", script)
        self.api.setComboboxSelectedIndex(self.ui, 9, 1, True)

    def set_body_names(self, names):
        self._names = names
        self.api.setComboboxItems(
            self.ui, 10, [f"Link {i + 1}: {n}" for i, n in enumerate(names)] + ["EEF payload"], 0, True
        )

    def poll(self):
        action = self.sim.callScriptFunction("takeAction", self.script)
        if action in ("upright", "bent"):
            q = np.zeros(7) if action == "upright" else np.degrees(HOME)
            for i, value in enumerate(q, 1):
                self.api.setSliderValue(self.ui, i, int(value), True)
        q = np.radians([self.api.getSliderValue(self.ui, i) for i in range(1, 8)])
        if self._last_q is None or not np.array_equal(q, self._last_q):
            for i, j in enumerate(self.joints):
                cyclic, interval = self.sim.getJointInterval(j)
                if not cyclic:
                    q[i] = np.clip(q[i], interval[0], interval[0] + interval[1])
                self.sim.setJointPosition(j, float(q[i]))
                self.sim.setJointTargetPosition(j, float(q[i]))
            self._last_q = q.copy()
            self.tests = {}
        payload = self.api.getSliderValue(self.ui, 8) / 10
        if payload != self._last_payload:
            self.tests = {}
            self._last_payload = payload
        return action, q, payload

    def draw(self, points, forces, parts, compensation):
        for h in (self.coms, self.weights, self.radius, self.joint_marker, self.axis):
            self.sim.addDrawingObjectItem(h, None)
        for p, f in zip(points, forces):
            self.sim.addDrawingObjectItem(self.coms, p.tolist())
            for segment in arrow_segments(p, self.force_scale * f):
                self.sim.addDrawingObjectItem(self.weights, segment.tolist())
        i = self.api.getComboboxSelectedIndex(self.ui, 9)
        body = self.api.getComboboxSelectedIndex(self.ui, 10)
        M = np.array(self.sim.getObjectMatrix(self.joints[i], self.sim.handle_world)).reshape(3, 4)
        self.sim.addDrawingObjectItem(self.joint_marker, M[:, 3].tolist())
        for segment in arrow_segments(M[:, 3], 0.15 * M[:, 2]):
            self.sim.addDrawingObjectItem(self.axis, segment.tolist())
        selected = np.zeros(7)
        if body < len(points):
            selected = parts[body]
            for segment in arrow_segments(M[:, 3], points[body] - M[:, 3]):
                self.sim.addDrawingObjectItem(self.radius, segment.tolist())
        table = "Joint    selected body gravity    total holding torque (Nm)\n"
        table += "\n".join(
            f"J{k + 1}:      {selected[k]:+8.3f}                  {compensation[k]:+8.3f}" for k in range(7)
        )
        self.api.setLabelText(self.ui, 20, table)
        weight = sum(np.linalg.norm(f) for f in forces)
        self.api.setLabelText(
            self.ui,
            21,
            f"Total gravity force magnitude: {weight:.2f} N | "
            "Holding torques exclude motor friction and acceleration terms.",
        )

    def report_test(self, mode, drift):
        self.tests[mode] = float(np.max(np.abs(drift)))
        text = " | ".join(f"{k}: max drift {v:.3f} deg" for k, v in self.tests.items())
        self.api.setLabelText(self.ui, 22, text)

    def close(self):
        if self.sim.isHandle(self.root):
            self.sim.removeObjects(self.sim.getObjectsInTree(self.root))
        for h, opacity in self.materials:
            if self.sim.isHandle(h):
                self.sim.setShapeColor(h, "@compound", self.sim.colorcomponent_transparency, opacity)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
