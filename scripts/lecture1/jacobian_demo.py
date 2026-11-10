"""KUKA Jacobian column demo: python scripts/lecture1/jacobian_demo.py

Green = measured EEF circular arc; the trace resets when the selected joint changes.
Blue = complete geometric reference circle, not a commanded full joint rotation.

Select a joint in the English panel. Yellow = joint origin; orange = joint-to-EEF
vector; cyan/magenta = linear/angular velocity for a unit or measured joint rate.
Angular arrow starts at the selected joint; linear arrow starts at the EEF.
All vectors are WORLD-frame geometric quantities, not Euler-angle rates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import os
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from lecture1.support.jacobian_geometry import eef_circle

HOME = np.radians([0, 30, 0, -60, 0, 30, 0])
AMPLITUDE, OMEGA = math.radians(15), 1.0
LINEAR_SCALE = .45  # seconds: 1 m/s -> a 0.45 m arrow
ANGULAR_SCALE = .18  # metres per rad/s: different units, different scale!


def geometric_jacobian(sim, joints, eef):
    p = np.array(sim.getObjectPosition(eef, sim.handle_world))
    J = np.zeros((6, len(joints)))
    for i, joint in enumerate(joints):
        M = np.array(sim.getObjectMatrix(joint, sim.handle_world)).reshape(3, 4)
        J[:3, i] = np.cross(M[:, 2], p-M[:, 3])
        J[3:, i] = M[:, 2]
    return J


def arrow_segments(origin, vector):
    """Three line segments: shaft plus arrowhead; no arrow for a zero vector."""
    length = np.linalg.norm(vector)
    if length < 1e-5:
        return []
    direction = vector / length
    side = np.cross(direction, [0, 0, 1] if abs(direction[2]) < .9 else [0, 1, 0])
    side /= np.linalg.norm(side)
    tip = origin + vector
    size = min(.04, length*.25)
    return [np.r_[origin, tip]] + [np.r_[tip, tip-size*direction+s*.5*size*side] for s in (-1, 1)]


def main(duration=None):
    client = RemoteAPIClient(host=os.getenv('COPPELIASIM_HOST', 'localhost'),
                             port=int(os.getenv('COPPELIASIM_PORT', '23000')))
    sim, ui_api = client.require('sim'), client.require('simUI')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop simulation before running this demo')
    robot = sim.getObject('/LBRiiwa14R820', {'noError': True})
    if robot == -1:
        robot = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir) +
                              '/models/robots/non-mobile/KUKA LBR iiwa 14 R820.ttm')
    path = sim.getObjectAlias(robot, 1)
    joints = [sim.getObject(path+'/joint', {'index': i}) for i in range(7)]
    eef = sim.getObject(path+'/connection')
    demo = sim.getScript(sim.scripttype_simulation, robot)
    if demo != -1:
        sim.setScriptInt32Param(demo, sim.scriptintparam_enabled, 0)
    for joint, q in zip(joints, HOME):
        sim.setJointMode(joint, sim.jointmode_dynamic, 0)
        sim.setIntProperty(joint, 'dynCtrlMode', sim.jointdynctrl_position)
        sim.setJointPosition(joint, float(q))
        sim.setJointTargetPosition(joint, float(q))
    # 4.10's legacy colorcomponent_transparency API stores OPACITY: 1=opaque.
    materials = []
    for shape in sim.getObjectsInTree(robot, sim.sceneobject_shape, 0):
        ok, opacity = sim.getShapeColor(shape, '@compound', sim.colorcomponent_transparency)
        if ok:
            materials.append((shape, opacity))
            sim.setShapeColor(shape, '@compound', sim.colorcomponent_transparency,
                              [min(value, .7) for value in opacity])
    # Remove only drawing geometry owned by earlier runs of our teaching demos.
    for name in ('/JacobianDemoVisuals', '/DH_FK_Demo'):
        old = sim.getObject(name, {'noError': True})
        if old != -1:
            sim.removeObjects(sim.getObjectsInTree(old))
    visuals = sim.createDummy(.001)
    sim.setObjectAlias(visuals, 'JacobianDemoVisuals')
    options = ''.join(f'<item>Joint {i+1}: column {i+1}</item>' for i in range(7))
    ui = ui_api.create(f'''<ui title="Jacobian | One joint, one column" closeable="false" layout="vbox">
<label text="Select one joint. Other joints hold their target angles."/>
<combobox id="1">{options}</combobox>
<checkbox id="2" text="Freeze target (measured joint velocity settles to zero)"/>
<checkbox id="5" text="Unit joint rate +1 rad/s: show Jacobian column (uncheck for measured rate)" checked="true"/>
<label text="YELLOW sphere: selected joint | ORANGE: joint-to-EEF vector r | CYAN at EEF: v | MAGENTA at joint: omega"/>
<label text="r = p_EEF - p_joint; v_i = (z_i cross r) * rate; omega_i = z_i * rate"/>
<label text="World frame. Arrows show ONLY the selected joint contribution, not the total EEF velocity."/>
<label text="GREEN: measured EEF arc; cleared when switching joints. EEF on the axis: position stays fixed."/>
<label text="BLUE: full reference circle, not the commanded joint range."/>
<label text="Arrow scales: linear 0.45 m per m/s; angular 0.18 m per rad/s. Sphere = joint origin."/>
<label id="4" text="Starting..."/><checkbox id="3" text="Exit"/></ui>''')
    ui_api.setComboboxSelectedIndex(ui, 1, 1, True)  # Joint 2 is easy to see.
    flags = sim.drawing_lines | sim.drawing_local | sim.drawing_painttag | sim.drawing_overlay
    drawings = [sim.addDrawingObject(flags, 6, 0, visuals, n, color)
                for n, color in [(3, [1,.35,0]),
                                 (3, [0,1,1]), (3, [1,0,1])]]
    trace = sim.addDrawingObject(flags,4,0,visuals,2000,[.15,1.,.2])
    circle_drawing = sim.addDrawingObject(flags,2,0,visuals,180,[.2,.45,1.])
    reference_circle = None
    previous_eef = None
    points = [sim.addDrawingObject(sim.drawing_spherepoints | sim.drawing_local | sim.drawing_painttag |
                                  sim.drawing_overlay, size, 0, visuals, count, color)
              for size, count, color in [(.009,7,[.8,.8,.8]), (.022,1,[1,.8,0]), (.012,1,[1,1,1])]]
    selected, phase_start, frozen_phase, next_report = 1, 0., 0., 0.
    sim.setStepping(True)
    try:
        sim.startSimulation()
        while not ui_api.getCheckboxValue(ui, 3):
            t = sim.getSimulationTime()
            if duration is not None and t >= duration:
                break
            index = ui_api.getComboboxSelectedIndex(ui, 1)
            if index != selected:
                selected, phase_start, frozen_phase = index, t, 0.
                sim.addDrawingObjectItem(trace,None)
                previous_eef = None
                reference_circle = None
            if not ui_api.getCheckboxValue(ui, 2):
                frozen_phase = t-phase_start
            else:
                phase_start = t-frozen_phase
            target = HOME.copy()
            target[selected] += AMPLITUDE*math.sin(OMEGA*frozen_phase)
            for joint, q in zip(joints, target):
                sim.setJointTargetPosition(joint, float(q))
            sim.step()
            J = geometric_jacobian(sim, joints, eef)
            qdot = sim.getJointVelocity(joints[selected])
            unit_rate = bool(ui_api.getCheckboxValue(ui, 5))
            rate = 1.0 if unit_rate else qdot
            v, omega = J[:3, selected]*rate, J[3:, selected]*rate
            p = np.array(sim.getObjectPosition(eef, sim.handle_world))
            if previous_eef is not None:
                sim.addDrawingObjectItem(trace,np.r_[previous_eef,p].tolist())
            previous_eef = p.copy()
            M = np.array(sim.getObjectMatrix(joints[selected], sim.handle_world)).reshape(3, 4)
            joint_origin, z = M[:,3], M[:,2]
            if reference_circle is None:
                reference_circle = eef_circle(joint_origin,z,p)
                sim.addDrawingObjectItem(circle_drawing,None)
                if len(reference_circle)>1:
                    edges = np.column_stack([reference_circle[:-1],reference_circle[1:]])
                    sim.addDrawingObjectItem(circle_drawing | sim.handleflag_addmultiple,edges.ravel().tolist())
            segments = [arrow_segments(joint_origin, p-joint_origin),
                        arrow_segments(p, LINEAR_SCALE*v),
                        arrow_segments(joint_origin, ANGULAR_SCALE*omega)]
            for handle in points:
                sim.addDrawingObjectItem(handle, None)
            origins = [sim.getObjectPosition(joint, sim.handle_world) for joint in joints]
            for origin in origins:
                sim.addDrawingObjectItem(points[0], origin)
            sim.addDrawingObjectItem(points[1], joint_origin.tolist())
            sim.addDrawingObjectItem(points[2], p.tolist())
            for handle, items in zip(drawings, segments):
                sim.addDrawingObjectItem(handle, None)
                for item in items:
                    sim.addDrawingObjectItem(handle, item.tolist())
            if sim.getSimulationTime() >= next_report:
                text = (f'Joint {selected+1} | measured qdot = {qdot:+.3f} rad/s | arrow rate = {rate:+.3f} rad/s\n'
                        f'J_v[:,i] = {J[:3,selected].round(3)} m/rad\n'
                        f'J_w[:,i] = {J[3:,selected].round(3)}\n'
                        f'v_i = {v.round(3)} m/s | omega_i = {omega.round(3)} rad/s')
                ui_api.setLabelText(ui, 4, text)
                next_report = sim.getSimulationTime()+.1
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stopSimulation()
        sim.setStepping(False)
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(.05)
        for handle in drawings + points + [trace,circle_drawing]:
            sim.removeDrawingObject(handle)
        sim.removeObjects([visuals])
        for shape, opacity in materials:
            if sim.isHandle(shape):
                sim.setShapeColor(shape, '@compound', sim.colorcomponent_transparency, opacity)
        ui_api.destroy(ui)


if __name__ == '__main__':
    main()
