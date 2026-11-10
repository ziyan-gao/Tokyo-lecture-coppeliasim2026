"""Move/rotate the green cylinder with CoppeliaSim's default tools; Ctrl+C/Stop exits.
Python computes the applied wrench and a first-order low-pass filter.
Graphs compare raw / filtered SENSOR force [N] and torque [Nm].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

import numpy as np
from lecture2.support.robot import Robot, arrow_segments
from shared.common import connect, clear_scene, _shape, trans, pose, stepped_simulation
from lecture2.support.force_sensor import physics, FilteredWrenchGraphs

DT = .005
TOOL_MASS, TOOL_LENGTH = .5, .12
Q_DOWN = np.radians([0, 30, 0, -60, 0, 90, 0])
DRAG_STIFFNESS, MAX_FORCE = 200., 20.       # N/m, N
ROTATION_STIFFNESS, MAX_TORQUE = 4., 2.    # Nm/rad, Nm
FILTER_TAU = .08  # s; approximate cutoff 1/(2*pi*tau) = 2 Hz.


def low_pass_filter(previous, sample, dt=DT, tau=FILTER_TAU):
    """First-order LPF: y += dt/(tau+dt) * (x-y); initialize from the first sample."""
    sample = np.asarray(sample,float)
    if previous is None:
        return sample.copy()
    return previous + dt/(tau+dt)*(sample-previous)


def setup(sim):
    """KUKA flange -> physical F/T sensor -> blue cylinder."""
    clear_scene(sim)
    h = Robot(sim, Q_DOWN).h
    h['root'] = sim.createDummy(.001)
    sim.setObjectAlias(h['root'], 'ForceLab')
    h['ft'] = sim.createForceSensor(0, [0, 1, 1, 0, 0], [.05, 1, 1, 0, 0])
    sim.setObjectAlias(h['ft'], 'ftSensor')
    sim.setObjectParent(h['ft'], h['flange'], True)
    sim.setObjectPose(h['ft'], sim.getObjectPose(h['eef'], sim.handle_world), sim.handle_world)
    h['tool'] = _shape(sim, sim.primitiveshape_cylinder, [.06, .06, TOOL_LENGTH],
        [.2, .45, .9], h['ft'], trans(z=TOOL_LENGTH/2), 'tool',
        dynamic=True, respondable=True, mass=TOOL_MASS)
    sim.setIntProperty(h['tool'], 'respondableMask', 0xFF00)
    return h


def limit(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


def spring_wrench(sim, tool_pose, hand_pose):
    """Virtual linear/angular springs; input poses and output wrench use WORLD axes."""
    displacement = np.array(hand_pose[:3]) - tool_pose[:3]
    axis, angle = sim.getRotationAxis(tool_pose, hand_pose)  # WORLD axis, radians
    force = limit(DRAG_STIFFNESS * displacement, MAX_FORCE)  # F = K dx
    torque = limit(ROTATION_STIFFNESS * angle * np.array(axis), MAX_TORQUE)
    return force, torque


def draw_force(sim, drawing, h, force_s):
    # Drawing is attached to the cylinder, so express the sensor force in tool axes.
    vector = limit(.012 * (pose(sim, h['ft'], h['tool'])[:3, :3] @ force_s), .28)
    sim.addDrawingObjectItem(drawing, None)
    for segment in arrow_segments(np.zeros(3), vector):
        sim.addDrawingObjectItem(drawing, segment.tolist())


def main():
    _, sim = connect()
    h = setup(sim)
    hand = _shape(sim, sim.primitiveshape_cylinder, [.068, .068, .125], [.05, .8, .2],
                  h['root'], pose(sim, h['tool']), 'Drag_Cylinder')
    sim.setObjectInt32Param(hand, sim.shapeintparam_wireframe, 1)
    graphs = FilteredWrenchGraphs(sim, h['root'])
    navigation = sim.getNavigationMode()
    tool_flags = sim.getObjectProperty(h['tool'])
    sim.setObjectProperty(h['tool'], tool_flags & ~sim.objectproperty_selectable)
    sim.setObjectSelection([hand])
    sim.setNavigationMode(sim.navigation_objectshift | sim.navigation_clickselection |
                          sim.navigation_camerazoomwheel | sim.navigation_camerarotatemiddlebutton)
    print('Move / Rotate green Drag_Cylinder with the default toolbar. Release mouse to unload.')
    print(f'Raw = pale curves; LPF = saturated curves. tau={FILTER_TAU:g} s (~2 Hz); arrow remains raw.')
    try:
        with physics(sim, DT), stepped_simulation(sim), graphs:
            drawing = sim.addDrawingObject(sim.drawing_lines | sim.drawing_local |
                sim.drawing_overlay, 5, 0, h['tool'], 3, [.95, 0, .15])
            deadline, step = time.perf_counter(), 0
            filtered = None
            while sim.getSimulationState() != sim.simulation_stopped:
                if sim.getSimulationState() == sim.simulation_paused:
                    time.sleep(.03)
                    deadline = time.perf_counter()
                    continue
                tool_pose = sim.getObjectPose(h['tool'], sim.handle_world)
                force_world, torque_world = np.zeros(3), np.zeros(3)
                if sim.getInt32Param(sim.intparam_mouse_buttons) & 1:
                    hand_pose = sim.getObjectPose(hand, sim.handle_world)
                    force_world, torque_world = spring_wrench(sim, tool_pose, hand_pose)
                else:
                    sim.setObjectPose(hand, tool_pose, sim.handle_world)
                sim.addForceAndTorque(h['tool'], force_world.tolist(), torque_world.tolist())
                sim.step()
                valid, force, torque = sim.readForceSensor(h['ft'])
                if valid & 1:
                    # Filter all six measured components in SENSOR coordinates.
                    raw = np.r_[force,torque]
                    filtered = low_pass_filter(filtered,raw)
                    graphs.update(raw,filtered)
                    draw_force(sim, drawing, h, np.array(force))
                    if step % 100 == 0:
                        print('F raw / LPF [N]:', np.round(raw[:3],3), np.round(filtered[:3],3),
                              '| T raw / LPF [Nm]:', np.round(raw[3:],3), np.round(filtered[3:],3))
                step += 1
                deadline = max(deadline + DT, time.perf_counter() - DT)
                time.sleep(max(0., deadline - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        sim.setNavigationMode(navigation)
        sim.setObjectProperty(h['tool'], tool_flags)
    print('Stopped. Raw / LPF force and torque graphs remain available.')


if __name__ == '__main__':
    main()
