"""Minimal KUKA joint-position control through the ZMQ remote API.

Setup (CoppeliaSim 4.10):
  1. Open a scene. The bundled KUKA model is loaded automatically if absent.
  2. Leave simulation stopped. The bundled robot demo script is disabled below.
  3. Run: python scripts/lecture1/kuka_zmq_control.py

Draw:   python scripts/lecture1/kuka_zmq_control.py --draw

Install: pip install coppeliasim-zmqremoteapi-client
Docker:  set COPPELIASIM_HOST=host.docker.internal
If using another model alias, change ROBOT_PATH below.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROBOT_PATH = "/LBRiiwa14R820"

# 1. Connect to CoppeliaSim.
client = RemoteAPIClient(
    host=os.environ.get("COPPELIASIM_HOST", "localhost"),
    port=int(os.environ.get("COPPELIASIM_PORT", "23000")),
)
sim = client.require("sim")
if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop the simulation before running this demo.")

# 2. Obtain the seven joint handles from the robot's hierarchy.
# 'index' distinguishes joints sharing the alias 'joint', from base to tool.
robot = sim.getObject(ROBOT_PATH, {"noError": True})
if robot == -1:
    resources = sim.getStringParam(sim.stringparam_resourcesdir)
    model_path = resources + "/models/robots/non-mobile/KUKA LBR iiwa 14 R820.ttm"
    robot = sim.loadModel(model_path)  # Path is on the simulator host, not Docker.
    ROBOT_PATH = sim.getObjectAlias(robot, 1)
    print(f"Loaded KUKA model: {ROBOT_PATH}")
joints = [sim.getObject(f"{ROBOT_PATH}/joint", {"index": i}) for i in range(7)]
for i, joint in enumerate(joints, start=1):
    print(f"Joint {i}: handle={joint}, path={sim.getObjectAlias(joint, 1)}")

# The joints must use dynamic position control for target-position commands.
for joint in joints:
    sim.setJointMode(joint, sim.jointmode_dynamic, 0)
    sim.setIntProperty(joint, "dynCtrlMode", sim.jointdynctrl_position)
# Keep the model's existing actuator limits and motion-profile settings.

# Each target is a function of SIMULATION time:
# q_i(t) = q_i(0) + A_i * sin(OMEGA * t), in radians.
AMPLITUDES = [math.radians(a) for a in [90, 90, 90, -90, 90, 90, 0]]
OMEGA = 0.1 # rad/s; use 0.1 for a slower sine wave (period = 2*pi/OMEGA).
DURATION = 10.0  # simulated seconds

# Disable the bundled autonomous demo while Python controls the robot.
demo_script = sim.getScript(sim.scripttype_simulation, robot)
if demo_script != -1:
    sim.setScriptInt32Param(demo_script, sim.scriptintparam_enabled, 0)

# Remove Graphs left by the earlier version of this demo.
for name in ('End_effector_XYZ_world', 'End_effector_orientation_world'):
    old = sim.getObject(f"{ROBOT_PATH}/{name}", {"noError": True})
    if old != -1:
        sim.removeObjects(sim.getObjectsInTree(old))

# Optional drawing objects: a world-space trail and a moving local XYZ frame.
drawings = []
if "--draw" in sys.argv:
    end_effector = sim.getObject(f"{ROBOT_PATH}/connection")
    line_type = sim.drawing_lines | sim.drawing_painttag
    trail = sim.addDrawingObject(line_type, 3, 0, -1, 10000, [0.1, 0.8, 1.0])
    drawings.append(trail)
    previous_position = sim.getObjectPosition(end_effector, sim.handle_world)
    # Local drawing coordinates follow the end-effector automatically.
    for axis, color in zip(([.15,0,0], [0,.15,0], [0,0,.15]),
                           ([1,0,0], [0,1,0], [0,0,1])):
        frame_axis = sim.addDrawingObject(line_type | sim.drawing_local,
                                         4, 0, end_effector, 1, color)
        sim.addDrawingObjectItem(frame_axis, [0,0,0] + axis)
        drawings.append(frame_axis)


# 3. Enable stepping before starting the simulation.
sim.setStepping(True)
try:
    # Hold the current pose at startup, instead of using stale model targets.
    initial = [sim.getJointPosition(joint) for joint in joints]
    for joint, angle in zip(joints, initial):
        sim.setJointTargetPosition(joint, angle)
    sim.startSimulation()
    next_report = 0.0
    while sim.getSimulationTime() < DURATION:
        # 4. Read simulation time, then compute and send joint targets.
        t = sim.getSimulationTime()
        targets = [q0 + amplitude * math.sin(OMEGA * t)
                   for q0, amplitude in zip(initial, AMPLITUDES)]
        for joint, target in zip(joints, targets):
            sim.setJointTargetPosition(joint, target)

        # 5. Advance exactly one simulation step, then repeat the control loop.
        sim.step()
        if drawings:
            position = sim.getObjectPosition(end_effector, sim.handle_world)
            sim.addDrawingObjectItem(trail, previous_position + position)
            previous_position = position
        if t >= next_report:
            actual_deg = [round(math.degrees(sim.getJointPosition(j)), 1) for j in joints]
            print(f"t = {sim.getSimulationTime():.2f} s | actual angles = {actual_deg}")
            next_report = t + 1.0
        time.sleep(0.02)  # Presentation pacing; the sine uses simulation time.
except (KeyboardInterrupt, EOFError):
    print("\nDemo interrupted.")
finally:
    sim.stopSimulation()
    sim.setStepping(False)
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.05)
    for drawing in drawings:
        sim.removeDrawingObject(drawing)
    print("Simulation stopped; stepping disabled.")
