"""Scene builders and displays: carrying, polishing and voxel maps."""

import math
from contextlib import ExitStack

import numpy as np

from lecture2.force_sensor import setup as sensor_setup
from shared.common import (
    CUBE_POS,
    CUBE_SIZE,
    TABLE_TOP,
    _dummy,
    _shape,
    build_scene,
    connect,
    get_q,
    pose,
    roty,
    rotz,
    set_q,
    stepped_simulation,
    to_sim,
    trans,
)

from . import camera as pc
from . import robot as kuka
from .force_sensor import ForceTrackingGraph, Payload, physics, settle_and_average
from .ompl_planner import make_collections
from .robot import IK
from .utils import OUT, save_report

MOUNT_POSE = trans(z=0.035) @ rotz(math.pi)
HANDLE_POSE = MOUNT_POSE @ trans(0.115, 0, 0) @ roty(math.pi / 2)
GOAL_OFFSET = np.array([0.0, -0.09, -0.025])
PAD_BOTTOM = 0.112


def _rigid_tool(sim, sensor, parts, name):
    """Group parts into one dynamic body, preserving the final part's frame."""
    tool = sim.groupShapes(parts, False)
    sim.setObjectAlias(tool, name)
    sim.setObjectParent(tool, sensor, True)
    sim.setBoolProperty(tool, "dynamic", True)
    sim.setBoolProperty(tool, "respondable", True)
    sim.setIntProperty(tool, "respondableMask", 0xFF00)
    return tool


class CarryScene:
    """Build a carried I-profile, red handle and receiving fixture."""

    goal_offset = GOAL_OFFSET

    @staticmethod
    def setup(sim):
        k = sensor_setup(sim)
        sim.setObjectColor(k["ft"], 0, sim.colorcomponent_ambient_diffuse, [0.05, 0.75, 0.95])
        sim.removeObjects([k["tool"]])
        parts = []
        # Every part becomes one rigid body under the real F/T sensor.
        for alias, size, T, color, mass, kind in [
            ("mount", [0.07, 0.07, 0.07], MOUNT_POSE, [0.17, 0.23, 0.30], 0.25, sim.primitiveshape_cuboid),
            ("upper_flange", [0.14, 0.10, 0.025], trans(z=0.10), [0.52, 0.72, 0.20], 0.25, sim.primitiveshape_cuboid),
            ("web", [0.035, 0.10, 0.055], trans(z=0.14), [0.52, 0.72, 0.20], 0.25, sim.primitiveshape_cuboid),
            ("lower_flange", [0.14, 0.10, 0.025], trans(z=0.18), [0.52, 0.72, 0.20], 0.25, sim.primitiveshape_cuboid),
            ("red_handle", [0.024, 0.024, 0.14], HANDLE_POSE, [0.85, 0.06, 0.04], 0.05, sim.primitiveshape_cylinder),
        ]:
            parts.append(_shape(sim, kind, size, color, k["ft"], T, alias, dynamic=True, respondable=True, mass=mass))
        # groupShapes takes the LAST part's frame. Use the mount frame, not the
        # cylinder frame: the handle centre is then (+.115, 0, 0) along body +X.
        k["tool"] = _rigid_tool(sim, k["ft"], parts[1:] + parts[:1], "carried_object_and_handle")
        # Fixed anchor follows the actual physical body, not its desired position.
        T = pose(sim, k["ft"]) @ HANDLE_POSE
        k["handle"] = _dummy(sim, "physical_handle", k["tool"], np.linalg.inv(pose(sim, k["tool"])) @ T, size=0.001)
        k["hand"] = _shape(
            sim, sim.primitiveshape_cylinder, [0.03, 0.03, 0.145], [0.95, 0.15, 0.05], k["root"], T, "Drag_Handle"
        )
        sim.setObjectInt32Param(k["hand"], sim.shapeintparam_wireframe, 1)
        sim.setObjectProperty(k["tool"], sim.getObjectProperty(k["tool"]) & ~sim.objectproperty_selectable)
        return k

    @staticmethod
    def goal_fixture(sim, k, rest):
        goal = rest + GOAL_OFFSET
        T = pose(sim, k["eef"])
        T[:3, 3] = goal
        root = _dummy(sim, "Goal_socket", k["root"], T, size=0.001)
        # The lower flange slides between two rails, with a small seating clearance.
        for alias, size, offset in [
            ("base", [0.23, 0.17, 0.025], trans(z=0.22)),
            ("left_rail", [0.025, 0.15, 0.07], trans(-0.095, 0, 0.18)),
            ("right_rail", [0.025, 0.15, 0.07], trans(0.095, 0, 0.18)),
        ]:
            _shape(sim, sim.primitiveshape_cuboid, size, [0.46, 0.51, 0.57], root, offset, alias, respondable=True)
        k["goal_marker"] = _shape(
            sim, sim.primitiveshape_cuboid, [0.145, 0.105, 0.005], [0.1, 0.7, 0.9], root, trans(z=0.201), "goal_region"
        )
        sim.setShapeColor(k["goal_marker"], None, sim.colorcomponent_transparency, [0.35])
        return goal

    @staticmethod
    def force_at_handle(sim, k, force):
        """Apply WORLD force at the physical handle, including its moment about tool COM."""
        T = pose(sim, k["tool"])
        com = T[:3, :3] @ np.array(sim.getProperty(k["tool"], "centerOfMass")) + T[:3, 3]
        point = np.array(sim.getObjectPosition(k["handle"], sim.handle_world))
        sim.addForceAndTorque(k["tool"], force.tolist(), np.cross(point - com, force).tolist())

    @staticmethod
    def drawing(sim, parent):
        return sim.addDrawingObject(sim.drawing_lines, 5, 0, parent, 3, [0.95, 0.06, 0.02])


class PolishingScene:
    """Build the physical pad and a flat or spherical workpiece."""

    pad_bottom = PAD_BOTTOM

    @staticmethod
    def setup(sim, curved=False):
        k = sensor_setup(sim)
        sim.removeObjects([k["tool"]])
        shaft = _shape(
            sim,
            sim.primitiveshape_cylinder,
            [0.035, 0.035, 0.10],
            [0.2, 0.3, 0.4],
            k["ft"],
            trans(z=0.05),
            "polishing_shaft",
            dynamic=True,
            respondable=True,
            mass=0.3,
        )
        diameter = 0.04 if curved else 0.075
        pad = _shape(
            sim,
            sim.primitiveshape_cylinder,
            [diameter, diameter, 0.012],
            [0.95, 0.45, 0.08],
            k["ft"],
            trans(z=0.106),
            "polishing_pad",
            dynamic=True,
            respondable=True,
            mass=0.15,
        )
        k["tool"] = _rigid_tool(sim, k["ft"], [pad, shaft], "polishing_tool")
        # Soft engine contact, not an artificial applied contact force.
        sim.setFloatArrayProperty(k["tool"], "mujoco.solref", [0.08, 1.0])
        sim.setFloatArrayProperty(k["tool"], "mujoco.solimp", [0.5, 0.8, 0.01, 0.5, 2.0])
        sim.setFloatArrayProperty(k["tool"], "mujoco.friction", [0.15, 0.005, 0.0001])
        T = pose(sim, k["ft"])
        bottom = T[:3, :3] @ np.array([0, 0, PAD_BOTTOM]) + T[:3, 3]
        k["surface_z"] = bottom[2] - 0.008
        k["clearance"] = 0.008
        if curved:
            k["surface_radius"] = 0.18
            k["surface_center"] = bottom - np.array([0, 0, k["clearance"] + k["surface_radius"]])
            k["surface"] = _shape(
                sim,
                sim.primitiveshape_spheroid,
                [0.36] * 3,
                [0.26, 0.58, 0.72],
                k["root"],
                trans(*k["surface_center"]),
                "curved_workpiece",
                respondable=True,
            )
        else:
            k["surface"] = _shape(
                sim,
                sim.primitiveshape_cuboid,
                [0.34, 0.34, 0.035],
                [0.26, 0.58, 0.72],
                k["root"],
                trans(bottom[0], bottom[1], k["surface_z"] - 0.0175),
                "workpiece",
                respondable=True,
            )
        sim.setFloatArrayProperty(k["surface"], "mujoco.solref", [0.08, 1.0])
        sim.setFloatArrayProperty(k["surface"], "mujoco.solimp", [0.5, 0.8, 0.01, 0.5, 2.0])
        size = [0.40, 0.40, 0.20] if curved else [0.39, 0.39, 0.10]
        z = k["surface_center"][2] - 0.10 if curved else k["surface_z"] - 0.085
        _shape(
            sim,
            sim.primitiveshape_cuboid,
            size,
            [0.22, 0.25, 0.29],
            k["root"],
            trans(bottom[0], bottom[1], z),
            "support",
            respondable=True,
        )
        return k


class PolishingLab:
    def __init__(self, dt):
        self.dt, self.log = dt, []
        self.path_start = None
        self.resources = ExitStack()

    def __enter__(self):
        try:
            self.client, self.sim = connect()
            sim = self.sim
            self.handles = k = PolishingScene.setup(sim)
            self.sensor = k["ft"]
            self.gravity = np.array(sim.getProperty(sim.handle_scene, "gravity"))
            T = pose(sim, k["tool"], self.sensor)
            com = T[:3, :3] @ np.array(sim.getProperty(k["tool"], "centerOfMass")) + T[:3, 3]
            self.payload = Payload(sim.getFloatProperty(k["tool"], "mass"), com)
            self.graph = ForceTrackingGraph(sim, k["root"])
            self.resources.enter_context(physics(sim, self.dt))
            self.resources.enter_context(stepped_simulation(sim))
            self.resources.enter_context(self.graph)
            settle_and_average(sim, self.sensor)
            self.controller = kuka.Robot.from_handles(sim, k).controller(self.dt)
            self.rest = self.controller.position().copy()
            self.trace = sim.addDrawingObject(sim.drawing_lines, 3, 0, k["root"], 10000, [0.1, 0.95, 0.25])
            self.previous_contact = None
            self.start = sim.getSimulationTime()
            return self
        except BaseException:
            self.resources.close()
            raise

    def elapsed(self):
        return self.sim.getSimulationTime() - self.start

    def observe(self, step, t, desired, raw, measured, delta, velocity, reference):
        self.graph.update(desired, measured)
        actual = self.controller.position()
        if step % 10 == 0 and measured > 1.0:
            contact = np.r_[actual[:2], self.handles["surface_z"] + 0.001]
            if self.previous_contact is not None:
                self.sim.addDrawingObjectItem(self.trace, np.r_[self.previous_contact, contact].tolist())
            self.previous_contact = contact
        self.log.append(np.r_[t, desired, raw, measured, delta, velocity, reference, actual])
        if step % 200 == 0:
            print(f"t={t:.1f}s | Fd={desired:.1f} N | Fm={measured:.2f} N | delta={delta * 1000:.2f} mm", flush=True)

    def save(self, duration, mass, damping):
        log = np.asarray(self.log)
        if not len(log):
            return
        columns = ["t_s", "desired_N", "raw_normal_N", "measured_N", "delta_m", "delta_dot_m_s"]
        columns += [f"{name}_{a}" for name in ("reference_m", "actual_m") for a in "xyz"]
        settled = log[log[:, 0] >= 0.75 * duration]
        save_report(
            "polishing_admittance",
            {
                "samples": log,
                "columns": columns,
                "virtual_mass_kg": mass,
                "damping_Ns_m": damping,
                "stiffness_N_m": 0.0,
                "tangent_start_s": self.path_start,
                "tangent_start_condition": "Force error within max(1 N, 10% of desired) continuously for 0.3 s.",
                "final_force_error_N": float(log[-1, 1] - log[-1, 3]),
                "settled_force_rmse_N": float(np.sqrt(np.mean((settled[:, 1] - settled[:, 3]) ** 2))),
                "control": "M*delta_ddot+B*delta_dot=Fd-Fm; WORLD z_cmd=z_nominal-delta",
                "force_source": "Physical pad/workpiece contact measured by the F/T sensor; tool weight removed.",
                "note": "Low-speed normal-force estimate; inertial reactions are not explicitly compensated. Engine contact compliance differs from virtual controller stiffness K=0.",
            },
        )
        np.savetxt(OUT / "polishing_admittance.csv", log, delimiter=",", header=",".join(columns), comments="")
        from .utils import plot_polishing as plot

        plot(log, motion_start=self.path_start)

    def __exit__(self, *exc):
        return self.resources.__exit__(*exc)


def height_colors(points, lower, upper):
    # Fixed colour scale across all scans: blue -> cyan -> yellow -> orange/red.
    palette = np.array([[0.10, 0.28, 0.85], [0.05, 0.75, 0.85], [0.98, 0.85, 0.20], [0.90, 0.16, 0.08]])
    height = np.clip((points[:, 2] - lower) / (upper - lower), 0, 1)
    return np.column_stack([np.interp(height, np.linspace(0, 1, 4), palette[:, i]) for i in range(3)])


class OccupancyView:
    def __init__(self, sim, parent, resolution, lower, upper):
        self.sim, self.lower, self.upper = sim, lower, upper
        self.offset = np.zeros(3)  # Overlay the measured surfaces in world coordinates.
        # Native OcTree has no alpha property; use a separate drawing for display.
        flags = sim.drawing_cubepts | sim.drawing_itemcolors | sim.drawing_50percenttransparency
        self.drawing = sim.addDrawingObject(flags, resolution * 0.92, 0, parent, 200000)

    def refresh(self, grid, octree):
        points = grid.centres()
        colors = height_colors(points, self.lower, self.upper)
        count = grid.show(self.sim, octree, np.rint(colors * 255).astype(int))
        self.refresh_points(points)
        return count

    def refresh_points(self, points):
        """Display native Octree centres without changing its accumulated voxels."""
        colors = height_colors(points, self.lower, self.upper)
        count = len(points)
        self.sim.addDrawingObjectItem(self.drawing, None)
        if count:
            # cube point: position, identity quaternion, RGB; send all cells in one RPC.
            quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
            items = np.column_stack([points + self.offset, quaternion, colors])
            self.sim.addDrawingObjectItem(self.drawing | self.sim.handleflag_addmultiple, items.ravel().tolist())
        return count


OBSTACLES = [
    (0.24, 0.15, 0.12, 0.10, 0.12),
    (0.62, 0.32, 0.12, 0.12, 0.24),
    (0.64, -0.30, 0.12, 0.12, 0.40),
    (0.28, -0.31, 0.10, 0.12, 0.18),
    (0.64, 0.12, 0.10, 0.10, 0.46),
    (0.64, -0.10, 0.14, 0.10, 0.30),
]
PLACE_POSITION = np.array([0.43, -0.34, TABLE_TOP + CUBE_SIZE / 2])


class VoxelScene:
    def __init__(self, client, sim, resolution, grid=None, name="octomap"):
        self.client, self.sim = client, sim
        self.resolution, self.name = resolution, name
        self.h = h = build_scene(sim, verbose=False)
        sim.removeObjects([h.pop("cube")])
        self.obstacles = [
            _shape(
                sim,
                sim.primitiveshape_cuboid,
                [w, d, z],
                [0.63, 0.69, 0.75],
                h["root"],
                trans(x, y, TABLE_TOP + z / 2),
                f"scan_obstacle_{i + 1}",
                respondable=True,
            )
            for i, (x, y, w, d, z) in enumerate(OBSTACLES)
        ]
        self.camera = h["camera"]
        sim.setObjectParent(self.camera, h["frames"][-1], True)
        sim.setObjectMatrix(self.camera, to_sim(trans(0, 0.06, 0.03)), sim.handle_parent)
        self.tip_camera = np.linalg.inv(pose(sim, h["tip"])) @ pose(sim, self.camera)
        self.ik = IK(client, sim, h)
        self.grid = grid
        self.octree = sim.createOctree(resolution, 0, 1.0)
        sim.setObjectAlias(self.octree, name)
        sim.setObjectParent(self.octree, h["root"], True)
        sim.setBoolProperty(self.octree, "collidable", True)
        sim.setIntProperty(self.octree, "layer", 0)  # Separate translucent voxel display.
        self.render = sim.createCollection(0)
        # Render only physical obstacles: no robot, target, or map feedback.
        for obj in [h["table"], h["wall"]] + self.obstacles:
            sim.addItemToCollection(self.render, sim.handle_single, obj, 0)
        sim.setObjectInt32Param(self.camera, sim.visionintparam_entity_to_render, self.render)
        self.rgbd_camera = pc.Camera(sim, self.camera)
        self.K = self.rgbd_camera.intrinsics()
        self.far = sim.getObjectFloatParam(self.camera, sim.visionfloatparam_far_clipping)
        self.robot, self.truth = make_collections(sim, h)
        self.world_map = sim.createCollection(0)
        sim.addItemToCollection(self.world_map, sim.handle_single, self.octree, 0)
        for obj in self.obstacles:
            sim.addItemToCollection(self.truth, sim.handle_single, obj, 0)
        self.observations, self.final_q, self.final_cube = [], None, None

    def insert_target(self):
        self.h["cube"] = _shape(
            self.sim,
            self.sim.primitiveshape_cuboid,
            [CUBE_SIZE] * 3,
            [0.9, 0.1, 0.1],
            self.h["root"],
            trans(*CUBE_POS),
            "cube",
            respondable=True,
        )

    def save_map(self):
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "occupancy.npz", occupied=self.grid.centres(), free=self.grid.centres(False))
        self.grid.save(OUT / "occupancy.ot")
        pc.save_ply(OUT / "occupancy.ply", self.grid.centres())

    def finish(self, plan):
        sim, h = self.sim, self.h
        self.final_q = get_q(sim, h["joints"])
        self.final_cube = sim.getObjectMatrix(h["cube"], sim.handle_world)
        placed = pose(sim, h["cube"])[:3, 3]
        error = np.linalg.norm(placed - PLACE_POSITION)
        if error > 0.015:
            raise RuntimeError(f"Placement error too large: {error * 1000:.1f} mm")
        save_report(
            self.name + "_basics",
            dict(
                plan,
                voxel_m=self.resolution,
                observations=self.observations,
                camera_mount="eye-in-hand",
                T_tip_camera=self.tip_camera,
                extra_obstacles=len(self.obstacles),
                display_offset_m=[0, 0, 0],
                voxel_transparency=0.5,
                height_color_range_m=[TABLE_TOP, 0.75],
                target_present_during_mapping=False,
                target_position_world=CUBE_POS,
                pregrasp_position_world=np.array(CUBE_POS) + [0, 0, 0.16],
                place_position_world=PLACE_POSITION,
                final_cube_position_world=placed,
                placement_error_m=error,
                grasp_mode="ideal attachment, released for physical settling",
                status="pick and place complete",
                planner_obstacles="measured OcTree only",
                final_joint_rad=self.final_q,
                final_tip_position_world=pose(sim, h["tip"])[:3, 3],
                **self.mapping_metadata(),
            ),
        )

    def mapping_metadata(self):
        return {
            "unknown": "Unobserved cells retain probability 0.5; not assumed free.",
            "planning_unknown_policy": "Only occupied cells block planning; unknown space is permitted. Ground-truth audit precedes execution.",
            "backend": "OctoMap C++ OcTree via octomap-python; occupied leaves exported to CoppeliaSim OcTree.",
            "octomap_file": "occupancy.ot",
            "free_samples": "Centres of free OctoMap leaves; sizes may differ.",
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        sim, h = self.sim, self.h
        try:
            if self.final_q is not None:
                set_q(sim, h["joints"], self.final_q)
            if self.final_cube is not None:
                if not sim.isHandle(h["cube"]):
                    self.insert_target()
                sim.setBoolProperty(h["cube"], "dynamic", False)
                sim.setObjectMatrix(h["cube"], self.final_cube, sim.handle_world)
        finally:
            self.ik.close()
            sim.setObjectInt32Param(self.camera, sim.visionintparam_entity_to_render, -1)
            for collection in (self.render, self.robot, self.truth, self.world_map):
                sim.destroyCollection(collection)


class OctreeScene(VoxelScene):
    def __init__(self, client, sim, resolution):
        super().__init__(client, sim, resolution, name="octree")

    def points(self):
        # Octree pose is identity in WORLD, so local voxel centres are world coordinates.
        return np.asarray(self.sim.getOctreeVoxels(self.octree)).reshape(-1, 3)

    def save_map(self):
        OUT.mkdir(parents=True, exist_ok=True)
        points = self.points()
        np.savez_compressed(OUT / "octree_map.npz", occupied=points, voxel_m=self.resolution)
        pc.save_ply(OUT / "octree_map.ply", points)

    def mapping_metadata(self):
        return {
            "backend": "CoppeliaSim native Octree; accumulated RGB-D surface voxels",
            "map_file": "octree_map.npz",
            "map_frozen_after_scan": True,
            "unknown": "No distinction between unseen and observed-free space.",
            "planning_unknown_policy": "Only measured occupied voxels block planning. Ground-truth audit precedes execution.",
        }


class OctomapGrid:
    def __init__(self, resolution=0.02, lower=(0.05, -0.60, 0.20), upper=(0.90, 0.60, 0.80)):
        import octomap  # Optional dependency, loaded only for the probabilistic backend.

        self.resolution = resolution
        self.lower, self.upper = np.array(lower), np.array(upper)
        self.tree = octomap.OcTree(resolution)
        self.occupied, self.free = np.empty((0, 3)), np.empty((0, 3))

    def integrate(self, origin, points):
        points = np.asarray(points, dtype=np.float64)
        points = points[
            np.isfinite(points).all(axis=1) & np.all((points >= self.lower) & (points < self.upper), axis=1)
        ]
        # Native ray casting, log-odds updates, and inner-node updates happen in C++.
        self.tree.insertPointCloud(pointcloud=points, origin=np.asarray(origin, dtype=np.float64))
        occupied, free = [], []
        for leaf in self.tree.begin_leafs():
            centre = np.asarray(leaf.getCoordinate())
            if leaf.getOccupancy() >= self.tree.getOccupancyThres():
                # OctoMap can merge leaves. Expand them only for the fixed-size
                # display/collision export; the original octree stays hierarchical.
                size = leaf.getSize()
                n = max(1, round(size / self.resolution))
                axis = (np.arange(n) + 0.5) * self.resolution - size / 2
                offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
                occupied.extend(centre + offsets)
            else:
                free.append(centre)  # Leaf centres; free leaves may have different sizes.
        self.occupied = np.asarray(occupied).reshape(-1, 3)
        self.free = np.asarray(free).reshape(-1, 3)
        return self.occupied

    def centres(self, occupied=True):
        return self.occupied if occupied else self.free

    def show(self, sim, octree, colors=None):
        sim.removeVoxelsFromOctree(octree, 0, None)
        if len(self.occupied):
            sim.insertVoxelsIntoOctree(
                octree,
                2 if colors is not None else 0,
                self.occupied.ravel().tolist(),
                [245, 145, 35] if colors is None else colors.ravel().tolist(),
            )
        return len(self.occupied)

    def save(self, path):
        # .ot preserves probabilities and the complete hierarchical map.
        if not self.tree.write(str(path)):
            raise RuntimeError(f"Could not save OctoMap to {path}")


class OctomapScene(VoxelScene):
    def __init__(self, client, sim, resolution):
        super().__init__(client, sim, resolution, grid=OctomapGrid(resolution))


VIEWS = [
    ((0.40, 0.25, 0.85), (0.40, 0.30, TABLE_TOP)),  # Pick area.
    ((0.25, 0.30, 0.80), (0.45, 0.00, 0.45)),  # Wall, +Y face.
    ((0.55, 0.00, 0.90), (0.45, 0.00, TABLE_TOP)),  # Wall, top.
    ((0.25, -0.30, 0.80), (0.45, 0.00, 0.45)),  # Wall, -Y face.
    ((0.40, -0.25, 0.85), (0.40, -0.30, TABLE_TOP)),  # Place area.
]


def look_at(position, target, x_hint=(0.0, -1.0, 0.0)):
    """WORLD camera pose: optical +Z looks at target; +X follows a projected hint."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target - position
    length = np.linalg.norm(z)
    if length < 1e-12:
        raise ValueError("Camera position and target must differ.")
    z /= length
    hint = np.asarray(x_hint, float)
    x = hint - np.dot(hint, z) * z
    if np.linalg.norm(x) < 1e-8:
        hint = np.eye(3)[np.argmin(np.abs(z))]
        x = hint - np.dot(hint, z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = position
    return T


TOOL_LENGTH = 0.12
