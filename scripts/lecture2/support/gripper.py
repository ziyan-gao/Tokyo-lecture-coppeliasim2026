"""Reusable grippers; physical jaws use contact/friction, teaching jaws use explicit attachment."""

import math

import numpy as np

from shared.common import FINGER_CLOSED, FINGER_OPEN, _shape, pose, roty, to_sim, trans

from .robot import move_joints, move_linear, pose_from


def set_fingers(sim, h, opening):
    for f, sgn in zip(h["fingers"], (1, -1)):
        p = sim.getObjectPosition(f, sim.handle_parent)
        sim.setObjectPosition(f, [sgn * opening, p[1], p[2]], sim.handle_parent)


def gripper(sim, h, close, duration=0.5):
    dt = sim.getSimulationTimeStep()
    n = max(2, int(duration / dt))
    a, b = (FINGER_OPEN, FINGER_CLOSED) if close else (FINGER_CLOSED, FINGER_OPEN)
    for k in range(1, n + 1):
        set_fingers(sim, h, a + (b - a) * k / n)
        sim.step()


def attach(sim, h, obj):
    """Grasp = make the object static and parent it to the tip (what most teaching
    grippers and CoppeliaSim's own suction pads do)."""
    sim.setBoolProperty(obj, "dynamic", False)
    sim.setObjectParent(obj, h["tip"], True)
    sim.resetDynamicObject(obj)


def detach(sim, h, obj):
    """Release: give the object back to the physics engine."""
    sim.setObjectParent(obj, h["root"], True)
    sim.setBoolProperty(obj, "dynamic", True)
    sim.resetDynamicObject(obj)


def pick(sim, h, ik, p_obj, R, approach=0.15, on_step=None):
    """Joint move above the object, straight down, close, attach, straight up."""
    T_pre = pose_from(R, np.asarray(p_obj) + [0, 0, approach])
    q_pre = ik.config_for(T_pre)
    if q_pre is None:
        raise RuntimeError("pre-grasp pose not reachable")
    move_joints(sim, h, q_pre, 2.0, on_step)
    move_linear(sim, h, ik, pose_from(R, p_obj), 1.0, on_step)
    gripper(sim, h, close=True)
    attach(sim, h, h["cube"])
    move_linear(sim, h, ik, T_pre, 1.0, on_step)


RG2_MAX_OPENING = 0.1018  # native model's motor travel (m); do not stretch its mechanism
RG2_MOTOR_ZERO = -0.048


def _rg2_mount(sim, h):
    for alias in ("fingerL", "fingerR", "palm", "gripperMount"):
        sim.removeObjects([sim.getObject("/Lab/" + alias)])
    flange = h["frames"][-1]
    # A moving rigid anchor provides the upstream body for the sensor constraint.
    anchor = _shape(
        sim,
        sim.primitiveshape_cylinder,
        [0.06, 0.06, 0.02],
        [0.25, 0.28, 0.32],
        flange,
        trans(z=-0.01),
        "flangeMount",
        respondable=False,
    )
    # The welded interface overlaps the native base: it must not generate contact forces.
    # Non-respondable anchors still provide the rigid body for the F/T constraint.
    sim.setBoolProperty(anchor, "kinematic", True)
    ft = sim.createForceSensor(0, [0, 1, 1, 0, 0], [0.05, 100, 10, 0, 0])
    sim.setObjectAlias(ft, "RG2_mount_FT")
    sim.setObjectParent(ft, anchor, True)
    sim.setObjectMatrix(ft, to_sim(pose(sim, flange)), sim.handle_world)
    sim.setColorProperty(ft, "color.diffuse", [0.02, 0.85, 0.95])
    gripper = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir) + "/models/components/grippers/RG2.ttm")
    sim.setObjectParent(gripper, ft, True)
    sim.setObjectPose(gripper, [0, 0, 0, 0, 0, 0, 1], ft)  # EXACT same pose, no mounting offset
    script = sim.getScript(sim.scripttype_simulation, gripper)
    if script != -1:
        sim.setScriptInt32Param(script, sim.scriptintparam_enabled, 0)
    parts = {sim.getObjectAlias(o): o for o in sim.getObjectsInTree(gripper)}
    motor, centre = parts["openCloseJoint"], parts["centerJoint"]
    for joint, force in ((motor, 12.0), (centre, 30.0)):
        sim.setIntProperty(joint, "dynCtrlMode", sim.jointdynctrl_position)
        sim.setJointTargetForce(joint, force)
        sim.setJointTargetPosition(joint, sim.getJointPosition(joint))
    # Native finger contact centres define the TCP; keep it outside the internal joint chain.
    pads = [parts["leftTouch"], parts["rightTouch"]]
    tcp = np.mean([sim.getObjectPosition(p, flange) for p in pads], axis=0)
    sim.setObjectMatrix(h["tip"], to_sim(trans(*tcp)), sim.handle_parent)
    h.update(
        gripper=gripper,
        wristSensor=ft,
        rg2_motor=motor,
        rg2_center=centre,
        padSensors=[parts["leftForceSensor"], parts["rightForceSensor"]],
        rg2_pads=pads,
        max_opening=RG2_MAX_OPENING,
    )


def _rg2_command(sim, h, opening):
    sim.setJointTargetPosition(h["rg2_motor"], float(np.clip(opening, 0, RG2_MAX_OPENING) + RG2_MOTOR_ZERO))
    # This replaces the native script's centre-joint callback, without custom Lua.
    sim.setJointTargetPosition(h["rg2_center"], -sim.getJointPosition(h["rg2_motor"]) / 2)


def _rg2_gap(sim, h):
    return sim.getJointPosition(h["rg2_motor"]) - RG2_MOTOR_ZERO


PARALLEL_MAX_OPENING = 0.082


def _parallel_mount(sim, h, pad_depth=0.075, max_opening=PARALLEL_MAX_OPENING):
    for alias in ("fingerL", "fingerR", "palm", "gripperMount"):
        sim.removeObjects([sim.getObject("/Lab/" + alias)])
    flange = h["frames"][-1]
    palm = _shape(
        sim,
        sim.primitiveshape_cuboid,
        [max_opening + 0.048, 0.055, 0.035],
        [0.24, 0.28, 0.35],
        flange,
        trans(z=0.015),
        "Parallel_Gripper",
        respondable=True,
    )
    sim.setBoolProperty(palm, "kinematic", True)
    joints, sensors = [], []
    for sign in (-1, 1):
        joint = sim.createJoint(sim.joint_prismatic, sim.jointmode_dynamic, 0, [0.025, 0.012])
        sim.setObjectAlias(joint, "Jaw_Left" if sign < 0 else "Jaw_Right")
        sim.setObjectParent(joint, palm, False)
        sim.setObjectMatrix(
            joint, to_sim(trans(x=sign * 0.0105, z=pad_depth - 0.015) @ roty(sign * math.pi / 2)), sim.handle_parent
        )
        sim.setJointInterval(joint, False, [0.0, max_opening / 2])
        sim.setIntProperty(joint, "dynCtrlMode", sim.jointdynctrl_position)
        sim.setJointTargetForce(joint, 12.0)
        sim.setJointPosition(joint, max_opening / 2)
        sim.setJointTargetPosition(joint, max_opening / 2)
        carrier = _shape(
            sim,
            sim.primitiveshape_cuboid,
            [0.012] * 3,
            [0.3, 0.3, 0.3],
            joint,
            trans(),
            "Jaw_Carrier",
            dynamic=True,
            mass=0.04,
        )
        sensor = sim.createForceSensor(0, [0, 1, 1, 0, 0], [0.01, 1, 1, 0, 0])
        sim.setObjectParent(sensor, carrier, False)
        sim.setObjectMatrix(sensor, to_sim(roty(-sign * math.pi / 2)), sim.handle_parent)
        pad = _shape(
            sim,
            sim.primitiveshape_cuboid,
            [0.021, 0.024, 0.04],
            [0.12, 0.15, 0.18],
            sensor,
            trans(),
            "Jaw_Pad",
            dynamic=True,
            respondable=True,
            mass=0.04,
        )
        sim.setFloatArrayProperty(pad, "mujoco.friction", [1.0, 0.01, 0.001])
        sim.setIntProperty(pad, "mujoco.condim", 4)
        joints.append(joint)
        sensors.append(sensor)
    sim.setObjectMatrix(h["tip"], to_sim(trans(z=pad_depth)), sim.handle_parent)
    h.update(gripper=palm, parallel_joints=joints, padSensors=sensors, max_opening=max_opening)


def _parallel_command(sim, h, opening):
    for joint in h["parallel_joints"]:
        sim.setJointTargetPosition(joint, float(np.clip(opening, 0, h["max_opening"]) / 2))


def _parallel_gap(sim, h):
    return float(sum(sim.getJointPosition(j) for j in h["parallel_joints"]))


class RG2Gripper:
    MAX_OPENING = RG2_MAX_OPENING
    mount = staticmethod(_rg2_mount)
    command = staticmethod(_rg2_command)
    gap = staticmethod(_rg2_gap)


class ParallelGripper:
    MAX_OPENING = PARALLEL_MAX_OPENING
    mount = staticmethod(_parallel_mount)
    command = staticmethod(_parallel_command)
    gap = staticmethod(_parallel_gap)


class Gripper:
    """Bound physical gripper. install() creates it; command()/gap() operate it."""

    def __init__(self, sim, handles, kind="rg2"):
        if kind not in ("rg2", "parallel"):
            raise ValueError("Physical gripper must be rg2 or parallel.")
        self.sim, self.h = sim, handles
        self.backend = RG2Gripper if kind == "rg2" else ParallelGripper

    @classmethod
    def install(cls, sim, handles, kind="rg2", **options):
        instance = cls(sim, handles, kind)
        instance.backend.mount(sim, handles, **options)
        return instance

    def command(self, opening):
        self.backend.command(self.sim, self.h, opening)

    def gap(self):
        return self.backend.gap(self.sim, self.h)
