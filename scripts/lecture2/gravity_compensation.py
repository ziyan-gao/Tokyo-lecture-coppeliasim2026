"""Gravity compensation from COM Jacobians: run this file with CoppeliaSim stopped.

    docker compose run --rm lab python lecture2/gravity_compensation.py

Inputs (distinguish measurements from model parameters):
    joint angles q            sim.getJointPosition        -> where every link and its COM is
    MODEL mass m, COM c       properties 'mass', 'centerOfMass' of every dynamic shape
    MODEL gravity vector g    scene property 'gravity'
Use: the torque gravity puts on joint i is the sum over the links it carries,

    tau_load = sum_k Jv_COM,k(q)^T (m_k g),        Jv column i = z_i x (c_k - o_i)

and the motors must hold tau_comp = -tau_load. No inverse-dynamics library is used.
The panel lets you change q and a point payload, and runs two 1 s physics tests in torque mode:
zero torque (the arm falls) vs. tau_comp (the arm floats).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time


import numpy as np

from lecture2.support.robot import GravityPanel
from lecture2.support.utils import save_report
from lecture2.support.force_sensor import physics

FORCE_SCALE = .008  # metres per newton, for drawing only


def read_bodies(sim, robot, joints):
    """Every dynamic shape is one rigid body: (shape, mass, COM in the shape frame, indices of the
    joints that carry it). Visual meshes (non-dynamic) have no mass and are skipped."""
    bodies = []
    for shape in sim.getObjectsInTree(robot, sim.sceneobject_shape, 0):
        if not sim.getBoolProperty(shape, 'dynamic'):
            continue
        ancestors = []
        parent = sim.getObjectParent(shape)
        while parent != -1:
            if parent in joints:
                ancestors.append(joints.index(parent))
            parent = sim.getObjectParent(parent)
        if ancestors:
            bodies.append((shape, sim.getFloatProperty(shape, 'mass'),
                           np.array(sim.getProperty(shape, 'centerOfMass')), ancestors))
    return bodies


def gravity_load(sim, joints, bodies, eef, gravity, payload=0.):
    """Gravity's joint torques, plus what we draw: COM positions, weights, per-body torques."""
    matrices = [np.array(sim.getObjectMatrix(j, sim.handle_world)).reshape(3, 4) for j in joints]
    points, forces, contributions = [], [], []
    for shape, mass, local_com, ancestors in bodies:
        M = np.array(sim.getObjectMatrix(shape, sim.handle_world)).reshape(3, 4)
        com = M[:, :3] @ local_com + M[:, 3]
        Jv = np.zeros((3, len(joints)))
        for i in ancestors:  # only joints between the base and this body move its COM
            Jv[:, i] = np.cross(matrices[i][:, 2], com - matrices[i][:, 3])
        points.append(com)
        forces.append(mass * gravity)
        contributions.append(Jv.T @ forces[-1])
    if payload > 0:  # a point mass at the flange: every joint carries it
        p = np.array(sim.getObjectPosition(eef, sim.handle_world))
        Jv = np.column_stack([np.cross(M[:, 2], p - M[:, 3]) for M in matrices])
        points.append(p)
        forces.append(payload * gravity)
        contributions.append(Jv.T @ forces[-1])
    contributions = np.array(contributions)
    return contributions.sum(axis=0), points, forces, contributions


def compare_dynamics(panel, bodies, q_start, payload, compensate, seconds=1., on_step=None):
    # Torque is recomputed every 5 ms, rather than held for a 50 ms display step.
    with physics(panel.sim,.005,engine=panel.sim.physics_bullet):
        return _compare_dynamics(panel,bodies,q_start,payload,compensate,seconds,on_step)


def _compare_dynamics(panel, bodies, q_start, payload, compensate, seconds, on_step):
    """A finite, automatic physics test; never wait for user input while stepping."""
    sim, joints = panel.sim, panel.joints
    old_engine = sim.getInt32Param(sim.intparam_dynamic_engine)
    old_ctrl = [sim.getIntProperty(j, 'dynCtrlMode') for j in joints]
    old_force = [sim.getJointTargetForce(j) for j in joints]
    damping = .05  # small viscous damping, identical in both cases; no position feedback
    try:
        sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_bullet)
        for j, q in zip(joints, q_start):
            sim.setJointPosition(j, float(q))
            sim.setIntProperty(j, 'dynCtrlMode', sim.jointdynctrl_force)
            sim.setJointTargetForce(j, 0., True)
        sim.setStepping(True)
        sim.startSimulation()
        while sim.getSimulationTime() < seconds:
            # read -> compute -> write -> step
            gravity = np.array(sim.getProperty(sim.handle_scene, 'gravity'))
            load, points, forces, parts = gravity_load(sim, joints, bodies, panel.eef, gravity, payload)
            qdot = np.array([sim.getJointVelocity(j) for j in joints])
            torque = (-load if compensate else np.zeros(len(joints))) - damping * qdot
            for j, value in zip(joints, torque):
                sim.setJointTargetForce(j, float(value), True)
            if payload > 0:
                # the point payload is an external force at the EEF, not extra link mass
                last, com = bodies[-1][0], points[len(bodies) - 1]
                moment = np.cross(points[-1] - com, forces[-1])
                sim.addForceAndTorque(last, forces[-1].tolist(), moment.tolist())
            sim.step()
            panel.draw(points, forces, parts, -load)
            if on_step is not None:
                on_step(points, forces, torque)
            # Stepping determines time; no additional wall-clock delay is needed.
        final = np.array([sim.getJointPosition(j) for j in joints])
        drift = np.degrees(final - q_start)
        print('WITH compensation' if compensate else 'WITHOUT compensation',
              '| joint drift (deg):', drift.round(3), flush=True)
        return drift
    finally:
        sim.stopSimulation()
        sim.setStepping(False)
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(.03)
        sim.setInt32Param(sim.intparam_dynamic_engine, old_engine)
        for j, q, ctrl, force in zip(joints, q_start, old_ctrl, old_force):
            sim.setIntProperty(j, 'dynCtrlMode', ctrl)
            sim.setJointTargetForce(j, float(force), True)
            sim.setJointPosition(j, float(q))
            sim.setJointTargetPosition(j, float(q))


def main():
    if '--batch' in sys.argv:
        with GravityPanel(FORCE_SCALE) as panel:
            bodies = read_bodies(panel.sim, panel.robot, panel.joints)
            q = np.array([panel.sim.getJointPosition(j) for j in panel.joints])
            off = compare_dynamics(panel, bodies, q, 0., False)
            on = compare_dynamics(panel, bodies, q, 0., True)
            save_report('gravity_compensation', dict(q_rad=q,
                without_compensation_drift_deg=off, with_compensation_drift_deg=on,
                note='q is measured; masses/COMs/gravity are model parameters. No position PD.'))
        return
    with GravityPanel(FORCE_SCALE) as panel:
        sim = panel.sim
        bodies = read_bodies(sim, panel.robot, panel.joints)
        panel.set_body_names([sim.getObjectAlias(b[0]) for b in bodies])
        try:
            while True:
                action, q, payload = panel.poll()
                if action == 'exit':
                    break
                if action in ('release', 'compensate'):
                    drift = compare_dynamics(panel, bodies, q, payload, action == 'compensate')
                    panel.report_test(action, drift)
                gravity = np.array(sim.getProperty(sim.handle_scene, 'gravity'))
                load, points, forces, parts = gravity_load(sim, panel.joints, bodies, panel.eef, gravity, payload)
                panel.draw(points, forces, parts, -load)
            if on_step is not None:
                on_step(points, forces, torque)
                time.sleep(.08)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
