# Lecture 2 helper classes

Control laws and teaching sequences stay in the lecture scripts. This package contains ten Python files, including `__init__.py`.

| Module | Main interface | Responsibility |
|---|---|---|
| `robot.py` | `Robot`, `CartesianController`, `IK`, `GravityPanel` | Handles, joint commands, kinematics, trajectories and gravity UI |
| `camera.py` | `Camera`, `LivePointCloud` | RGB-D, calibration, cloud processing and display |
| `ompl_planner.py` | `OMPLPlanner` | Planning tasks, collision checks, path validation and tree display |
| `force_sensor.py` | `ForceSensor`, `Payload`, native graphs | Wrench data, tool-weight compensation and force curves |
| `gripper.py` | `Gripper`, `RG2Gripper`, `ParallelGripper` | Physical mounting/opening; explicit attachment for teaching jaws |
| `bin_cell.py` | `VisionCell`, `BinCell` | Robot-free RGB-D scene and RG2 picking cell |
| `scenes.py` | `CarryScene`, `PolishingScene`, `PolishingLab`, voxel-map scenes | Demo geometry, resource lifecycle and display |
| `grconvnet.py` | `GraspModel` | Network, pretrained weights and planar-grasp inference |
| `utils.py` | `save_report`, summary plots | On-demand output artifacts |

Typical bound APIs:

```python
robot = Robot.from_handles(sim, handles)  # Reuse a loaded robot.
rgb, depth = Camera(sim, camera_handle).rgbd()
force, torque = ForceSensor(sim, sensor_handle).read()  # Raw sensor-frame wrench.
robot.set_targets(q_target)
path = OMPLPlanner(client, sim, handles, collision_pairs).plan(q_start, q_goal)
```

`Robot.controller(dt)` provides Cartesian commands for the position-controlled arm. `Gripper.install(sim, handles, 'rg2')` mounts a physical gripper; `command()` and `gap()` operate it. The lecture script explicitly calls `sim.step()`.

`VisionCell` shares placement, caching and simulation setup with `BinCell`. Scene builders share rigid-tool mounting; native graphs share construction and viewer code. Native Octree requires no OctoMap binding; the optional probabilistic backend loads its library only when instantiated.
