"""BinCell scenes, cached YCB placement and physical grasp execution."""

import hashlib
import json
import math
import os
from contextlib import contextmanager

import numpy as np

from shared import ycb
from shared.common import (
    OUT_DIR,
    TABLE_TOP,
    _shape,
    build_scene,
    clear_scene,
    first,
    get_q,
    pose,
    rotx,
    rotz,
    set_q,
    stepped_simulation,
    to_sim,
    trans,
)

from .camera import Camera
from .force_sensor import read_wrench
from .gripper import Gripper, RG2Gripper
from .robot import IK, move_joints, move_linear, pose_from

OBJECTS = [
    "005_tomato_soup_can",
    "004_sugar_box",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "013_apple",
    "017_orange",
    "014_lemon",
    "018_plum",
]
PROXY = {"011_banana": "coacd"}
BIN = np.array([0.45, 0.22])  # bin centre (x, y), world
PLACE = np.array([0.45, -0.25])  # the empty bin
BIN_INNER, BIN_HEIGHT, WALL = (0.30, 0.26), 0.09, 0.01
FLOOR = TABLE_TOP + WALL  # bin floor height
CAM_HEIGHT, CAM_FOV, CAM_RES, CAM_NEAR, CAM_FAR = 0.70, math.radians(50.0), 224, 0.2, 1.2


class VisionCell:
    def __init__(
        self,
        client,
        sim,
        seed=1,
        new_heap=False,
        objects=None,
        bin_inner=BIN_INNER,
        placement_margin=0.005,
        engine=None,
        bullet_version=None,
    ):
        self.client, self.sim = client, sim
        self.engine = sim.physics_mujoco if engine is None else engine
        self.bullet_version = bullet_version
        self.bin_inner = np.asarray(bin_inner, float)
        self.placement_margin = placement_margin
        self.object_names = list(OBJECTS if objects is None else objects)
        self.rng = np.random.default_rng(seed)
        self.heap_file = os.path.join(os.path.dirname(OUT_DIR), "data", "scenes", f"bin_heap_seed{seed}.json")
        if self.object_names != OBJECTS:
            suffix = hashlib.sha256(",".join(self.object_names).encode()).hexdigest()[:8]
            self.heap_file = self.heap_file.replace(".json", f"_{suffix}.json")
        self.simConvex, self.simOpenMesh = client.require("simConvex"), client.require("simOpenMesh")
        self.ik = None
        self.h = self._build()
        self.camera = Camera(sim, self.h["binCamera"])
        self.loaded = not new_heap and self._load_heap(self.h["root"])
        if not self.loaded:
            self.h["objects"] = self._drop_objects(self.h["root"])
            self._report_overlaps()

    def _build(self):
        sim = self.sim
        clear_scene(sim)
        root = sim.createDummy(0.001)
        sim.setObjectAlias(root, "Lab")
        table = _shape(
            sim,
            sim.primitiveshape_cuboid,
            [1.4, 1.4, 0.05],
            [0.55, 0.42, 0.30],
            root,
            trans(0.2, 0, TABLE_TOP - 0.025),
            "table",
            respondable=True,
        )
        return self._add_bins_and_camera({"root": root, "table": table})

    def _heap_key(self):
        key = {"objects": self.object_names, "proxy": {n: PROXY.get(n, "hull") for n in self.object_names}}
        if not np.allclose(self.bin_inner, BIN_INNER) or self.placement_margin != 0.005:
            key.update(bin_inner=self.bin_inner.tolist(), placement_margin=self.placement_margin)
        return key

    def _load_heap(self, root):
        """Spawn the objects straight at their cached, settled poses. False if there is no valid cache."""
        if not os.path.exists(self.heap_file):
            return False
        with open(self.heap_file) as f:
            data = json.load(f)
        if data["key"] != self._heap_key():
            print("heap cache is for other objects/proxies: building a new heap")
            return False
        self.h["objects"] = {}
        for name in self.object_names:
            body = ycb.spawn(
                self.sim, self.simConvex, self.simOpenMesh, name, root, PROXY.get(name, "hull"), tuple(BIN), FLOOR
            )
            self.sim.setObjectMatrix(body, data["poses"][name], self.sim.handle_world)
            self.sim.setBoolProperty(body, "detectable", True)  # seen by the pad's proximity sensor
            self.h["objects"][name] = body
        print(f"loaded the settled heap from {os.path.relpath(self.heap_file, os.path.dirname(OUT_DIR))}")
        return True

    def _save_heap(self):
        os.makedirs(os.path.dirname(self.heap_file), exist_ok=True)
        poses = {n: self.sim.getObjectMatrix(b, self.sim.handle_world) for n, b in self.h["objects"].items()}
        with open(self.heap_file, "w") as f:
            json.dump({"key": self._heap_key(), "poses": poses}, f, indent=1)
        print(f"saved the settled heap to {os.path.relpath(self.heap_file, os.path.dirname(OUT_DIR))}")

    def _add_bins_and_camera(self, h):
        sim = self.sim
        self._make_bin(h["root"], BIN, "bin")
        self._make_bin(h["root"], PLACE, "placeBin")
        cam = sim.createVisionSensor(
            2, [CAM_RES, CAM_RES, 0, 0], [CAM_NEAR, CAM_FAR, CAM_FOV, 0.05, 0, 0, 0.9, 0.9, 0.92, 0, 0]
        )
        sim.setObjectAlias(cam, "binCamera")
        sim.setObjectParent(cam, h["root"], True)
        sim.setObjectMatrix(cam, to_sim(trans(*BIN, TABLE_TOP + CAM_HEIGHT) @ rotx(math.pi)), sim.handle_world)
        h["binCamera"] = cam
        return h

    def _make_bin(self, root, centre, name):
        """A static box open at the top: floor + 4 walls."""
        (lx, ly), z0 = self.bin_inner, TABLE_TOP
        parts = [([lx + 2 * WALL, ly + 2 * WALL, WALL], (0, 0, WALL / 2))]
        for s in (-1, 1):
            parts.append(([WALL, ly + 2 * WALL, BIN_HEIGHT], (s * (lx + WALL) / 2, 0, BIN_HEIGHT / 2)))
            parts.append(([lx, WALL, BIN_HEIGHT], (0, s * (ly + WALL) / 2, BIN_HEIGHT / 2)))
        for k, (size, (dx, dy, dz)) in enumerate(parts):
            _shape(
                self.sim,
                self.sim.primitiveshape_cuboid,
                size,
                [0.45, 0.5, 0.55],
                root,
                trans(centre[0] + dx, centre[1] + dy, z0 + dz),
                f"{name}{k}",
                respondable=True,
            )

    def _drop_objects(self, root):
        """Objects above the bin, randomly turned, NOT overlapping: each one goes to the lowest
        spot (of 40 random ones) where its bounding box is clear of the objects already placed.
        They fall a few cm and settle into a heap."""
        sim, bodies, boxes = self.sim, {}, []
        (lx, ly) = self.bin_inner
        for name in self.object_names:
            body = ycb.spawn(
                sim, self.simConvex, self.simOpenMesh, name, root, PROXY.get(name, "hull"), tuple(BIN), FLOOR
            )
            T = pose(sim, body)
            c = T[:3, 3]
            R = rotz(self.rng.uniform(0, 2 * math.pi)) @ rotx(self.rng.choice([0.0, math.pi / 2]))  # stand or lie
            sim.setObjectMatrix(body, to_sim(trans(*c) @ R @ trans(*-c) @ T), sim.handle_world)
            lo, hi = world_aabb(sim, body)
            half = (hi - lo)[:2] / 2
            best = None
            for _ in range(40):
                xy = BIN + self.rng.uniform(-1, 1, 2) * np.maximum([lx / 2, ly / 2] - half - self.placement_margin, 0)
                z = FLOOR + 0.005  # bottom of the box: on the floor, or on top of what is below
                for blo, bhi in boxes:
                    if np.all(np.abs(xy - (blo[:2] + bhi[:2]) / 2) < half + (bhi - blo)[:2] / 2 + 0.005):
                        z = max(z, bhi[2] + 0.005)
                if best is None or z < best[1]:
                    best = xy, z
            xy, z = best
            shift = np.r_[xy - (lo[:2] + hi[:2]) / 2, z - lo[2]]
            sim.setObjectPosition(
                body, (np.array(sim.getObjectPosition(body, sim.handle_world)) + shift).tolist(), sim.handle_world
            )
            boxes.append((lo + shift, hi + shift))
            sim.setBoolProperty(body, "detectable", True)  # seen by the pad's proximity sensor
            bodies[name] = body
        return bodies

    def _report_overlaps(self):
        """Exact check (proxy shapes) of the start state: object-object and object-robot."""
        sim, names = self.sim, list(self.h["objects"])
        bad = [
            f"{a[4:]}/{b[4:]}"
            for i, a in enumerate(names)
            for b in names[i + 1 :]
            if first(sim.checkCollision(self.h["objects"][a], self.h["objects"][b]))
        ]
        if "joints" in self.h:
            robot = sim.createCollection(0)
            sim.addItemToCollection(robot, sim.handle_tree, self.h["joints"][0], 0)
            bad += [f"{a[4:]}/robot" for a in names if first(sim.checkCollision(self.h["objects"][a], robot))]
            sim.destroyCollection(robot)
        print(f"{len(names)} objects placed, overlapping at the start: {', '.join(bad) if bad else 'none'}")

    @contextmanager
    def running(self):
        """Selected physics engine, one simulation for the whole run:
        stopping it would put the objects back in the air. Lets the heap settle first."""
        sim = self.sim
        old_dt = sim.getFloatProperty(sim.handle_scene, "timeStep")
        old_dyn_dt = sim.getFloatProperty(sim.handle_scene, "dynamicsStepSize")
        sim.setFloatProperty(sim.handle_scene, "timeStep", 0.01)
        sim.setFloatProperty(sim.handle_scene, "dynamicsStepSize", 0.005)
        old = (
            sim.getIntArrayProperty(sim.handle_scene, "dynamicsEngine"),
            sim.getProperty(sim.handle_scene, "mujoco.cone"),
        )
        if self.engine == sim.physics_bullet and self.bullet_version is not None:
            version = 0 if self.bullet_version == 278 else self.bullet_version
            sim.setIntArrayProperty(sim.handle_scene, "dynamicsEngine", [self.engine, version])
        else:
            sim.setInt32Param(sim.intparam_dynamic_engine, self.engine)
        self.actual_engine = sim.getIntArrayProperty(sim.handle_scene, "dynamicsEngine")
        print(f"Physics engine read-back [engine, version]: {self.actual_engine}", flush=True)
        if self.engine == sim.physics_mujoco:
            sim.setProperty(sim.handle_scene, "mujoco.cone", 1)
        try:
            with stepped_simulation(sim):
                if self.loaded:
                    self.wait(0.2)
                else:
                    self.wait(2.0)  # let the heap settle
                    self._put_back_strays()
                    self._save_heap()  # and remember it
                yield self
        finally:
            sim.setIntArrayProperty(sim.handle_scene, "dynamicsEngine", old[0])
            sim.setProperty(sim.handle_scene, "mujoco.cone", old[1])
            sim.setFloatProperty(sim.handle_scene, "timeStep", old_dt)
            sim.setFloatProperty(sim.handle_scene, "dynamicsStepSize", old_dyn_dt)
            if self.ik is not None:
                self.ik.close()

    def _put_back_strays(self, rounds=3):
        """Objects that bounced out of the bin are dropped again onto the middle of the heap."""
        sim = self.sim
        for _ in range(rounds):
            out = [b for b in self.h["objects"].values() if not in_box(self.com(b), BIN, 0.0, self.bin_inner)]
            if not out:
                return
            top = max((world_aabb(sim, b)[1][2] for b in self.h["objects"].values() if b not in out), default=FLOOR)
            for b in out:
                lo, hi = world_aabb(sim, b)
                xy = BIN + self.rng.uniform(-0.03, 0.03, 2)
                shift = np.r_[xy - (lo[:2] + hi[:2]) / 2, top + 0.01 - lo[2]]
                sim.setObjectPosition(
                    b, (np.array(sim.getObjectPosition(b, sim.handle_world)) + shift).tolist(), sim.handle_world
                )
                sim.resetDynamicObject(b)
                top = hi[2] + shift[2]
            print(f"put {len(out)} stray object(s) back into the bin")
            self.wait(2.0)

    def wait(self, seconds):
        for _ in range(int(seconds / self.sim.getSimulationTimeStep())):
            self.sim.step()

    def com(self, body):
        T = pose(self.sim, body)
        return (T @ np.append(self.sim.getProperty(body, "centerOfMass"), 1))[:3]

    def where(self):
        """{'bin': [...], 'placeBin': [...], 'table/floor': [...]} from the true object positions."""
        out = {}
        for name, body in self.h["objects"].items():
            p = self.com(body)
            spot = (
                "bin"
                if in_box(p, BIN, -0.01, self.bin_inner)
                else "placeBin"
                if in_box(p, PLACE, -0.01, self.bin_inner)
                else "table/floor"
            )
            out.setdefault(spot, []).append(name[4:])
        return out


def world_aabb(sim, shape):
    """World-axis bounding box (lo, hi) of a shape: the 8 corners of its own box, in the world."""
    size, p = sim.getShapeBB(shape)
    local = np.eye(4)
    local[:3, :3], local[:3, 3] = quat_to_R(p[3:]), p[:3]
    T = pose(sim, shape) @ local
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(size) / 2
    W = (T @ np.c_[corners, np.ones(8)].T)[:3].T
    return W.min(axis=0), W.max(axis=0)


def quat_to_R(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def in_box(p, centre, margin=0.02, size=BIN_INNER):
    """Inside a bin's interior, at least `margin` from the walls."""
    return all(abs(p[i] - centre[i]) < size[i] / 2 - margin for i in (0, 1))


PICK_OBJECTS = ["008_pudding_box", "009_gelatin_box", "010_potted_meat_can"]
GRASP_DEPTH, APPROACH = 0.03, 0.15


class BinCell(VisionCell):
    """VisionCell plus one RG2-equipped robot, grasp execution and delivery checks.

    Reuses the robot-free cell's object placement, cache and simulation lifecycle.
    """

    def __init__(
        self, client, sim, seed=1, new_heap=False, engine=None, friction=1.5, mass_scale=0.5, bullet_version=283
    ):
        self.max_opening = RG2Gripper.MAX_OPENING
        super().__init__(
            client,
            sim,
            seed=seed,
            new_heap=new_heap,
            objects=PICK_OBJECTS,
            bin_inner=(0.42, 0.38),
            placement_margin=0.065,
            engine=sim.physics_bullet if engine is None else engine,
            bullet_version=bullet_version,
        )
        self.friction = friction
        self.mass_scale = mass_scale
        self.object_masses = {}
        for name, body in self.h["objects"].items():
            mass = sim.getFloatProperty(body, "mass") * mass_scale
            # 'inertia' stores I/m: keep it unchanged so physical I scales with mass.
            sim.setFloatProperty(body, "mass", mass)
            self.object_masses[name] = sim.getFloatProperty(body, "mass")
        for shape in sim.getObjectsInTree(self.h["root"], sim.sceneobject_shape):
            sim.setFloatProperty(shape, "bullet.friction", friction)
            sim.setFloatProperty(shape, "bullet.frictionOld", friction)
            sim.setFloatProperty(shape, "bullet.restitution", 0.0)
            sim.setBoolProperty(shape, "bullet.stickyContact", False)

    def _build(self):
        sim = self.sim
        handles = build_scene(sim, verbose=False)
        for alias in ("cube", "wall", "placeMarker", "camera"):
            sim.removeObjects([sim.getObject("/Lab/" + alias)])
        self.jaw = Gripper.install(sim, handles, "rg2")
        self._add_bins_and_camera(handles)
        self.ik = IK(self.client, sim, handles)
        self.R_down = pose(sim, handles["tip"])[:3, :3]
        self.park = self.ik.config_for(pose_from(self.R_down, [*PLACE, TABLE_TOP + 0.35]))
        self.above_bin = self.ik.config_for(pose_from(self.R_down, [*BIN, TABLE_TOP + 0.35]))
        set_q(sim, handles["joints"], self.park)  # Park before introducing any objects.
        return handles

    def capture(self):
        if np.max(np.abs(np.asarray(get_q(self.sim, self.h["joints"])) - self.park)) > 0.001:
            move_joints(self.sim, self.h, self.park, 1.5)
        self.sim.step()
        rgb, depth = self.camera.rgbd()
        return rgb, depth, pose(self.sim, self.h["binCamera"])

    def tool_pose(self, jaw, centre):
        """Tool down, its x axis (the closing direction of the fingers) along `jaw`."""
        x0 = self.R_down[:, 0]
        R = rotz(math.atan2(jaw[1], jaw[0]) - math.atan2(x0[1], x0[0]))[:3, :3] @ self.R_down
        return pose_from(R, centre)

    def plan(self, g):
        """Pre-grasp and grasp poses plus the pre-grasp configuration, or None if out of reach.
        A parallel gripper turned by 180 deg makes the same grasp: try both (the wrist may reach one)."""
        z = max(
            g["top"][2] - g.get("depth", GRASP_DEPTH), FLOOR + 0.015
        )  # the finger ends (1 cm below the tip) stay off the floor
        for jaw in (g["jaw"], -g["jaw"]):
            T_pre = self.tool_pose(jaw, [g["top"][0], g["top"][1], g["top"][2] + APPROACH])
            q_pre = self.ik.config_for(T_pre, steps=60, seed=self.above_bin)
            T_grasp = self.tool_pose(jaw, [g["top"][0], g["top"][1], z])
            if q_pre is not None and self.ik.config_for(T_grasp, steps=60, seed=q_pre) is not None:
                return T_pre, T_grasp, q_pre, g["opening"]
        return None

    def set_gap(self, gap, duration=0.6):
        """Command pad separation for a fixed duration; zero squeezes until contact."""
        for _ in range(int(duration / self.sim.getSimulationTimeStep())):
            self.jaw.command(max(0.0, gap))
            self.sim.step()
        return self.jaw.gap()

    def pad_force(self):
        """Squeeze force felt by the two pad force sensors [N] (mean of both)."""
        return float(np.mean([np.linalg.norm(read_wrench(self.sim, s)[0]) for s in self.h["padSensors"]]))

    def execute(self, plan):
        sim, h = self.sim, self.h
        pre, grasp, q_pre, width = plan
        empty = np.array([read_wrench(sim, s)[0] for s in h["padSensors"]])
        opening = min(width + 0.02, self.max_opening)
        self.set_gap(opening, 0.4)
        move_joints(sim, h, q_pre, 1.5)
        move_linear(sim, h, self.ik, grasp, 1.0)
        gap = self.set_gap(0.0, 0.8)
        force = self.pad_force()
        if not -0.005 <= gap <= self.max_opening + 0.005:
            return False, f"INVALID_DYNAMICS: gap {gap:.3f} m"
        if gap < 0.004 or gap > opening - 0.004 or force < 2.0:
            # Retract before opening wide: opening down inside the bin can hit a wall.
            move_linear(sim, h, self.ik, pre, 0.8)
            self.set_gap(self.max_opening, 0.5)
            return False, f"No secure grasp: gap={gap:.3f} m, force={force:.1f} N"
        move_linear(sim, h, self.ik, pre, 1.2)
        if self.pad_force() < 2.0:
            self.set_gap(self.max_opening, 0.5)
            return False, "Object slipped during lift"
        move_joints(sim, h, self.park, 1.5)
        if self.pad_force() < 2.0:
            self.set_gap(self.max_opening, 0.5)
            return False, "Object dropped during transport"
        opened = self.set_gap(self.max_opening, 0.8)
        self.wait(0.8)
        after = np.array([read_wrench(sim, s)[0] for s in h["padSensors"]])
        residual = np.linalg.norm(after - empty, axis=1).max()
        released = opened >= self.max_opening - 0.005 and residual < 0.5
        if not released:
            return False, f"RELEASE_FAILED: gap={opened:.3f}, residual force={residual:.2f} N; stopped"
        return (
            True,
            f"Released: held gap={gap:.3f} m, force={force:.1f} N, open gap={opened:.3f} m, residual={residual:.2f} N",
        )
