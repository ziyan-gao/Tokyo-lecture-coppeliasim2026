# Lecture 2: From sensor data to robot actions

Teaching pattern: **READ → TRANSFORM → COMPUTE → COMMAND → VERIFY**.

Run one controller at a time with CoppeliaSim 4.10. Stop the previous demo first.
Local connections default to `localhost:23000`; Docker configures `host.docker.internal`.
Commands below run from the repository root.

| Entry | Sensor data | Example |
|---|---|---|
| `gravity_compensation.py` | Encoders, model masses/COMs, gravity | COM Jacobians and compensation torques |
| `force_sensor.py` | Raw sensor-frame force/torque | Drag/rotate the cylinder; native raw/filtered graphs |
| `admittance.py` | Weight-compensated world-frame force | Virtual mass, damping and stiffness |
| `cooperative_carry.py` | Handle force transmitted through a physical F/T sensor | Guide the carried I-profile towards a socket |
| `polishing_admittance.py` | Measured normal contact force | XY path plus normal force-error admittance, K=0 |
| `curved_polishing_admittance.py` | Force projected onto the known local surface normal | Spherical workpiece and rotating tool orientation |
| `vision_sensor.py` | RGB bytes and metric depth | Robot-free YCB scene; save current images |
| `rgbd_pointcloud.py` | RGB-D and camera calibration | Back-project coloured points into world coordinates |
| `bin_picking_grconvnet.py` | Depth, grasp predictions, finger forces | Clearance checks, physical RG2 grasp and delivery verification |
| `ompl_pick_place.py` | Camera localisation and collision geometry | One RRT transport across the green obstacle; finish holding the cube |
| `octree_basics.py` | Eye-in-hand RGB-D at five poses | Static native Octree, one RRTConnect transport, complete pick/place |

```bash
python scripts/lecture2/gravity_compensation.py
python scripts/lecture2/gravity_compensation.py --batch
python scripts/lecture2/force_sensor.py
python scripts/lecture2/admittance.py --interactive
python scripts/lecture2/admittance.py --stiffness 0 --mass 2 --damping 40
python scripts/lecture2/cooperative_carry.py --demo
python scripts/lecture2/polishing_admittance.py
python scripts/lecture2/curved_polishing_admittance.py
python scripts/lecture2/vision_sensor.py
python scripts/lecture2/rgbd_pointcloud.py
python scripts/lecture2/bin_picking_grconvnet.py --detect
python scripts/lecture2/bin_picking_grconvnet.py
python scripts/lecture2/ompl_pick_place.py RRT
python scripts/lecture2/octree_basics.py --voxel .02
```

## Modules

`support/` has 10 Python files, including the package marker. Reusable interfaces are
`Robot`, `Camera`, `OMPLPlanner`, `ForceSensor`, `Gripper`, `VisionCell` / `BinCell`, and
`GraspModel`. Demo-specific geometry/displays are in `scenes.py`; reports and optional
plots are in `utils.py`. See [the class guide](support/README.md).

There are no recording flags or recording modules. Generated images, point clouds,
plots and reports are created on demand under the ignored `scripts/out/` directory.

## Interpretation

- Raw F/T includes tool weight; admittance compensates it before using world-frame force.
- Polishing uses real simulated contact, not a synthesized contact-force signal.
- GR-ConvNet predicts planar grasps; depth segmentation and measured cloud clearance are post-processing.
- RG2 bin picking uses physical contact/friction. Map-demo cube grasps use ideal attachment followed by physical settling after release.
- Static Octree stores measured surface occupancy. OctoMap integrates occupied/free evidence using the native C++ library.
- Map-based planning blocks occupied cells; incomplete observations can still miss obstacles, so execution includes a separate geometry audit.

Dependencies and simulator models are listed in the repository [README](../../README.md).

## Reading the teaching code

The entry scripts expose the algorithms rather than forwarding whole pipelines:

- `rgbd_pointcloud.py`: sensor API, byte decoding, back-projection and world transform.
- `bin_picking_grconvnet.py`: detection, clearance checks and the continuous perception/action loop.
- `octree_basics.py`: scanning and the grasp/plan/transport/release sequence.
- `polishing_admittance.py`: CLI, tangent path, force feedback and normal admittance update.
- `curved_polishing_admittance.py`: surface normals and tool-orientation trajectory.

`support/force_sensor.py` groups small force/physics helpers; `support/utils.py` groups optional
summary figures. `VisionCell` constructs the robot-free YCB scene; `BinCell` adds robot/gripper
execution for learned bin picking. Both live in `support/bin_cell.py`.

### Bound interfaces

- `Robot.from_handles(sim, h).controller(dt)` reuses an existing position-controlled arm.
- `Camera(sim, handle).rgbd()` reads metric RGB-D.
- `ForceSensor(sim, handle, payload).read()` returns the raw sensor-frame wrench.
- `OMPLPlanner(client, sim, h, pairs).plan(start, goal)` binds collision scope and algorithm.
- `Gripper.install(sim, h, 'rg2')` creates the physical gripper; `command()` and `gap()` operate it.
- `GraspModel().predict(depth)` reuses loaded network weights.
