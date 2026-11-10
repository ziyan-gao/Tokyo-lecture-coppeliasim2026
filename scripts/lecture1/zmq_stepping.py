"""Minimal ZMQ remote API + stepping demo (CoppeliaSim 4.10).

Install: pip install coppeliasim-zmqremoteapi-client
Run:     python scripts/lecture1/zmq_stepping.py

Open CoppeliaSim first and leave the simulation stopped.
Creates a blue ball and a landing platform. Use a scene with gravity enabled.
Select /SteppingDemo and fit the view in CoppeliaSim to see the ball.
For Docker, set COPPELIASIM_HOST=host.docker.internal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 1. Connect to CoppeliaSim's ZMQ remote API server (default port: 23000).
host = os.environ.get("COPPELIASIM_HOST", "localhost")
port = int(os.environ.get("COPPELIASIM_PORT", "23000"))
client = RemoteAPIClient(host=host, port=port)
sim = client.require("sim")
print(f"Connected to {host}:{port}")

if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError("Stop the simulation in CoppeliaSim before running this demo.")

# 2. Prepare a simple free-fall scene (metres, kilograms).
# Remove only this demo's previous objects when running it again.
old_demo = sim.getObject("/SteppingDemo", {"noError": True})
if old_demo != -1:
    sim.removeObjects(sim.getObjectsInTree(old_demo))
root = sim.createDummy(0.001)
sim.setObjectAlias(root, "SteppingDemo")

platform = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.8, 0.8, 0.08], 0)
sim.setObjectAlias(platform, "landing_platform")
sim.setObjectParent(platform, root, True)
sim.setObjectPosition(platform, [0, 0, 0.04], sim.handle_world)
sim.setBoolProperty(platform, "dynamic", False)
sim.setBoolProperty(platform, "respondable", True)
sim.setShapeColor(platform, "", sim.colorcomponent_ambient_diffuse, [0.18, 0.22, 0.30])

ball = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.16, 0.16, 0.16], 0)
sim.setObjectAlias(ball, "blue_ball")
sim.setObjectParent(ball, root, True)
sim.setObjectPosition(ball, [0, 0, 1.2], sim.handle_world)
sim.computeMassAndInertia(ball, 1000)
sim.setFloatProperty(ball, "mass", 0.2)
sim.setBoolProperty(ball, "dynamic", True)
sim.setBoolProperty(ball, "respondable", True)
sim.setShapeColor(ball, "", sim.colorcomponent_ambient_diffuse, [0.08, 0.48, 0.95])
sim.setShapeColor(ball, "", sim.colorcomponent_specular, [0.8, 0.8, 0.8])


# 3. Enable stepping BEFORE starting: Python decides when time advances.
sim.setStepping(True)
try:
    sim.startSimulation()
    print(f"Simulation step size: {sim.getSimulationTimeStep():.3f} s")
    for step in range(1, 21):
        before = sim.getSimulationTime()
        input(f"t = {before:.3f} s | Press Enter for step {step}/20 ... ")
        # A robot control loop would read sensors and send commands here.
        sim.step()  # Advance exactly one simulation step, then wait again.
        after = sim.getSimulationTime()
        height = sim.getObjectPosition(ball, sim.handle_world)[2]
        print(f"  {before:.3f} -> {after:.3f} s | ball height = {height:.3f} m")
except (KeyboardInterrupt, EOFError):
    print("\nDemo interrupted.")
finally:
    # 4. Stop and release stepping even if the user presses Ctrl+C.
    sim.stopSimulation()
    sim.setStepping(False)
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.05)
    print("Simulation stopped; stepping disabled.")
