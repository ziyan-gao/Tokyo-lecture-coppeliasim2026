"""Continuously draw a tilted circle using the 7 DoF model's BUILT-IN simIK script.
Run with simulation stopped: python scripts/lecture1/simik_basics.py
Python only moves manipSphere and advances simulation; the model solves IK.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import os
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

PERIOD, RADIUS = 6., .09  # simulated seconds per circle; metres
TILT = math.radians(35)
client = RemoteAPIClient(host=os.getenv('COPPELIASIM_HOST', 'localhost'),
                         port=int(os.getenv('COPPELIASIM_PORT', '23000')))
sim = client.require('sim')
if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError('Stop simulation before running this demo')
robot = sim.getObject('/redundantRobot', {'noError': True})
if robot == -1:
    robot = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir) +
                          '/models/robots/non-mobile/7 DoF manipulator.ttm')
path = sim.getObjectAlias(robot, 1)
sphere = sim.getObject(path+'/manipSphere')
tip = sim.getObject(path+'/tip')
# Re-enable the built-in IK script disabled by our earlier version.
script = sim.getScript(sim.scripttype_simulation, robot)
sim.setScriptInt32Param(script, sim.scriptintparam_enabled, 1)
pose0 = sim.getObjectPose(sphere, sim.handle_world)  # XYZ + quaternion
for name in ('/simIK_Spiral', '/simIK_Circle'):
    old = sim.getObject(name, {'noError': True})
    if old != -1:
        sim.removeObjects(sim.getObjectsInTree(old))
trace_root = sim.createDummy(.001)
sim.setObjectAlias(trace_root, 'simIK_Circle')
dt = sim.getSimulationTimeStep()
trace = sim.addDrawingObject(sim.drawing_lines | sim.drawing_local |
    sim.drawing_painttag | sim.drawing_cyclic, 4, 0, trace_root, round(2*PERIOD/dt), [0,.8,1])

sim.setStepping(True)
try:
    sim.startSimulation()
    start = sim.getSimulationTime()
    previous = sim.getObjectPosition(tip, sim.handle_world)
    while True:
        angle = 2*math.pi*(sim.getSimulationTime()-start+dt)/PERIOD
        pose = pose0.copy()  # Keep the orientation quaternion unchanged.
        pose[0] += RADIUS*math.sin(angle)*math.sin(TILT)
        pose[1] += RADIUS*(math.cos(angle)-1)  # Circle starts at the initial pose.
        pose[2] += RADIUS*math.sin(angle)*math.cos(TILT)
        sim.setObjectPose(sphere, pose, sim.handle_world)
        sim.step()  # Built-in script solves IK; robot moves for one simulation step.
        actual = sim.getObjectPosition(tip, sim.handle_world)
        sim.addDrawingObjectItem(trace, previous+actual)
        previous = actual
        time.sleep(.01)
except KeyboardInterrupt:
    pass
finally:
    sim.stopSimulation()
    sim.setStepping(False)
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(.03)
print('Finished. The cyan line is the actual EEF trajectory.')
