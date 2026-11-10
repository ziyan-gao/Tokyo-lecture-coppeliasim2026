"""Real objects: spawn scanned YCB household objects (textured meshes) into the scene.

    docker compose run --rm lab python lecture1/ycb_objects.py               # kitchen table, CoACD mode by default
    docker compose run --rm lab python lecture1/ycb_objects.py raw show      # original meshes, no proxies
    docker compose run --rm lab python lecture1/ycb_objects.py raw show newton # compare Newton Dynamics
    docker compose run --rm lab python lecture1/ycb_objects.py hull          # every proxy a convex hull
    docker compose run --rm lab python lecture1/ycb_objects.py spheres       # every proxy a set of spheres
    docker compose run --rm lab python lecture1/ycb_objects.py coacd         # CoACD instead of V-HACD
    docker compose run --rm lab python lecture1/ycb_objects.py spheres show  # ... and leave the proxies visible
    docker compose run --rm lab python lecture1/ycb_objects.py 035_power_drill 048_hammer 013_apple
    docker compose run --rm lab python lecture1/ycb_objects.py spheres save  # build, then save the scene
    docker compose run --rm lab python lecture1/ycb_objects.py spheres load  # skip building: load it

The YCB object set (www.ycbbenchmarks.com, CC BY 4.0) is the standard benchmark of robot
grasping: 77 everyday objects, each scanned as a 16k-triangle mesh with a photo texture, and
with its real mass. Objects are downloaded on first use (~11 MB each) into scripts/data/ycb/.
Names: see ycb.MASS_G (or the list at https://ycb-benchmarks.s3.amazonaws.com/data/objects.json).

A scan is a "random" mesh: slow and unstable for dynamics (shape_types.py). So, as there:
  visual   the textured scan, static and not respondable: only drawn
  proxy    what the physics engine sees, with the real mass, hidden in layer 9:
           default  CoACD decomposition for every object, including nearly convex boxes
           hull     always the hull: the bowl's is a closed dome (the apple stays on top),
                    the banana's rocks on its curved belly and creeps across the table for ever
           coacd    default mode (pip install coacd): decomposes every object,
                    without a hull fallback; results are cached after the first run
           spheres  sphere decomposition (ycb.fit_spheres), a primitive compound: comes to rest
                    best of all, concave shapes need no special case; few spheres for compact
                    objects (apple 22), many for flat faces (cracker box 159) and thin walls
                    (bowl 210), which become a cage of 8 mm spheres with gaps in between
Raw mode skips mesh simplification, convexity analysis, and proxy generation.
The original textured mesh is both visible and dynamic; "raw show" keeps it visible.
MuJoCo and Newton Dynamics still use its convex hull internally for collisions.
Every run saves out/ycb_<mode>.png: the scans next to the proxies (one colour per CoACD piece).
With "show" the proxies stay visible in the scene; otherwise toggle layer 9 in CoppeliaSim.
"save" stores the built /Lab as a model (scripts/data/scenes/ycb_<mode>.ttm), "load" loads it instead
of building it. sim.saveModel / sim.loadModel exchange the model as bytes, so the file is written by
Python and it does not matter on which machine CoppeliaSim runs.

Making things come to rest (the script measures the drift of each object between 3 s and 5 s):
  * the right proxy (above), and a hull that is not too coarse (ycb.spawn)
  * MuJoCo: rolling friction, which needs condim 6 (ycb.spawn), and an elliptic friction cone
    (scene setting mujoco.cone = 1: the default pyramidal cone lets resting objects creep)
Bullet is the default engine, with friction 0.8 on objects and table.
Pass "mujoco" or "newton" to compare engines. Engine settings are
restored afterwards. MuJoCo and Newton use convex hulls for non-convex raw meshes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import colorsys
import math
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np

from shared import ycb
from shared.common import OUT_DIR, clear_scene, connect, rotx, rotz, stepped_simulation, to_sim, trans

PROXIES = {None: "hull / V-HACD pieces", "hull": "convex hull", "coacd": "CoACD pieces", "spheres": "spheres", "raw": "original mesh"}
ENGINES = ("bullet", "mujoco", "newton")
ENGINE = next((a for a in sys.argv[1:] if a in ENGINES), "bullet")
PROXY = next((a for a in sys.argv[1:] if a in PROXIES), "coacd")
FLAGS = {a for a in sys.argv[1:] if a in ("show", "save", "load")}
OBJECTS = [a for a in sys.argv[1:] if a not in PROXIES and a not in FLAGS and a not in ENGINES]
SCENE_FILE = os.path.join(os.path.dirname(ycb.DATA_DIR), "scenes",
                          "_".join(["ycb", PROXY or "auto"] + [n.split("_")[0] for n in OBJECTS]) + ".ttm")

BULLET_FRICTION = 0.8

TABLE_TOP = 0.02  # above the scene's floor (z = 0): coplanar faces flicker (z-fighting)
BOWL = "024_bowl"
AROUND = OBJECTS or ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
                     "006_mustard_bottle", "025_mug", "011_banana", "017_orange", "014_lemon", "018_plum"]
INTO_BOWL = [] if OBJECTS else ["013_apple"]  # the 16 cm bowl has room for one big fruit

client, sim = connect()
simConvex, simOpenMesh = (None, None) if PROXY == "raw" else (
    client.require("simConvex"), client.require("simOpenMesh"))


def build_table():
    """/Lab with a table and a camera looking at it from the front."""
    clear_scene(sim)
    root = sim.createDummy(0.001)
    sim.setObjectAlias(root, "Lab")
    table = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [1.0, 1.0, 0.04], 0)
    sim.setObjectAlias(table, "table")
    sim.setObjectParent(table, root, True)
    sim.setObjectPosition(table, [0, 0, TABLE_TOP - 0.02], sim.handle_world)
    sim.setShapeColor(table, "", sim.colorcomponent_ambient_diffuse, [0.55, 0.42, 0.30])
    sim.setBoolProperty(table, "respondable", True)
    camera = sim.createVisionSensor(2, [960, 600, 0, 0], [0.02, 4, math.radians(55), 0.05, 0, 0, 0.92, 0.92, 0.95, 0, 0])
    sim.setObjectAlias(camera, "camera")
    sim.setObjectParent(camera, root, True)
    T = trans(0, -0.75, TABLE_TOP + 0.53) @ rotx(-math.radians(125)) @ rotz(math.pi)  # image up = world up
    sim.setObjectMatrix(camera, to_sim(T), sim.handle_world)
    return root, camera


def placements():
    """name -> (xy, height of the lowest point): bowl in the middle, a ring around it, fruit above it."""
    place = {} if OBJECTS else {BOWL: ((0.0, 0.0), TABLE_TOP + 0.002)}
    for k, name in enumerate(AROUND):
        a = 2 * math.pi * k / len(AROUND)
        place[name] = ((0.3 * math.cos(a), 0.3 * math.sin(a)), TABLE_TOP + 0.002)
    for k, name in enumerate(INTO_BOWL):
        place[name] = ((0.01, 0.0), TABLE_TOP + 0.12 + 0.1 * k)
    return place


def spawn_raw(name, root, xy, z):
    """Import the original mesh directly, without creating collision proxies."""
    vertices, faces, uv, texture = ycb.load(name)
    body = ycb.create_textured_shape(sim, vertices, faces, uv, texture)
    # Parent immediately so a failed build is still removed by the next clear_scene.
    sim.setObjectParent(body, root, True)
    sim.setObjectAlias(body, name)
    sim.computeMassAndInertia(body, 1000.0)
    sim.setFloatProperty(body, "mass", ycb.MASS_G[name] / 1000)
    sim.setBoolProperty(body, "dynamic", True)
    sim.setBoolProperty(body, "respondable", True)
    sim.setIntProperty(body, "layer", ycb.VISIBLE_LAYER)
    if ENGINE == "mujoco":
        sim.setFloatArrayProperty(body, "mujoco.friction", [1.0, 0.01, 0.01])
        sim.setIntProperty(body, "mujoco.condim", 6)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    shift = [xy[0] - (lo[0] + hi[0]) / 2, xy[1] - (lo[1] + hi[1]) / 2, z - lo[2]]
    sim.setObjectPosition(body, list(np.add(sim.getObjectPosition(body, sim.handle_world), shift)), sim.handle_world)
    return body, len(faces)


def spawn_all(root):
    """Spawn every object at its place; returns name -> proxy handle."""
    if PROXY == "raw":
        print("\nRaw mode: original textured meshes; no simplification or proxies.", flush=True)
        if ENGINE in ("mujoco", "newton"):
            print(f"{ENGINE}: non-convex meshes use convex hulls internally; bowl cavities are not preserved.", flush=True)
        bodies = {}
        for name, (xy, z) in placements().items():
            started = time.time()
            body, count = spawn_raw(name, root, xy, z)
            bodies[name] = body
            print(f"{name:22s} {count:6d} triangles | {time.time() - started:.1f}s", flush=True)
        return bodies
    print("\nvisual: the textured scan, 16384 triangles each.  proxy: what the physics engine sees")
    print(f"{'object':22s} {'mass':>6s} {'convexity':>9s}   {'proxy':48s} {'time':>5s}")
    bodies = {}
    for name, (xy, z) in placements().items():
        t0 = time.time()
        body = bodies[name] = ycb.spawn(sim, simConvex, simOpenMesh, name, root, PROXY, xy, z)
        print(f"{name:22s} {sim.getFloatProperty(body, 'mass') * 1000:5.0f}g {ycb.spawn.convexity:9.2f}   "
              f"{shape_kind(body):20s} {ycb.spawn.detail:27s} {time.time() - t0:4.1f}s")
    return bodies


def save_scene(root):
    """The whole /Lab tree as a model file (saveModel needs a model base: /Lab becomes one)."""
    sim.setBoolProperty(root, "modelBase", True)
    os.makedirs(os.path.dirname(SCENE_FILE), exist_ok=True)
    with open(SCENE_FILE, "wb") as f:
        f.write(sim.saveModel(root))  # no file name: CoppeliaSim returns the model as bytes
    print(f"saved the scene: {os.path.relpath(SCENE_FILE, os.path.dirname(OUT_DIR))} "
          f"({os.path.getsize(SCENE_FILE) / 1e6:.1f} MB)")


def load_scene():
    """/Lab from the model file; returns root, camera and name -> proxy handle."""
    if not os.path.exists(SCENE_FILE):
        sys.exit(f"no saved scene {SCENE_FILE}: run once with 'save' (same mode and objects)")
    clear_scene(sim)
    t0 = time.time()
    with open(SCENE_FILE, "rb") as f:
        root = sim.loadModel(f.read())
    tree = sim.getObjectsInTree(root)
    camera = next(h for h in tree if sim.getObjectAlias(h) == "camera")
    bodies = {sim.getObjectAlias(h): h for h in tree if sim.getObjectAlias(h) in ycb.MASS_G}
    print(f"\nloaded {len(bodies)} objects from {os.path.relpath(SCENE_FILE, os.path.dirname(OUT_DIR))} "
          f"in {time.time() - t0:.1f} s")
    return root, camera, bodies


def shape_kind(h):
    kind = "primitive" if sim.getBoolProperty(h, "primitive") else "convex" if sim.getBoolProperty(h, "convex") else "random"
    return ("compound " if sim.getBoolProperty(h, "compound") else "") + kind


def com(h):
    """World position of the centre of mass (an object's frame need not be at its centre)."""
    T = np.vstack([np.array(sim.getObjectMatrix(h, sim.handle_world)).reshape(3, 4), [0, 0, 0, 1]])
    return (T @ np.append(sim.getProperty(h, "centerOfMass"), 1))[:3]


def show_proxies(bodies, on):
    """Swap the layers of proxy and scan: the proxies visible (one colour per convex piece for CoACD), or hidden."""
    if PROXY == "raw":
        # The original scan is the body itself; there is no proxy or visual child.
        for body in bodies.values():
            sim.setIntProperty(body, "layer", ycb.VISIBLE_LAYER)
        return
    for k, body in enumerate(bodies.values()):
        sim.setIntProperty(body, "layer", ycb.VISIBLE_LAYER if on else ycb.HIDDEN_LAYER)
        sim.setIntProperty(sim.getObjectChild(body, 0), "layer", ycb.HIDDEN_LAYER if on else ycb.VISIBLE_LAYER)
        if PROXY == "coacd" and sim.getBoolProperty(body, "compound"):
            # @compound addresses component materials without ungrouping the rigid body.
            _, current = sim.getShapeColor(body, "@compound", sim.colorcomponent_ambient_diffuse)
            colors = []
            for piece in range(len(current) // 3):
                colors.extend(colorsys.hsv_to_rgb((k * 0.17 + piece * 0.61803398875) % 1, 0.70, 0.95))
            sim.setShapeColor(body, "@compound", sim.colorcomponent_ambient_diffuse, colors)
        else:
            rgb = colorsys.hsv_to_rgb((k * 0.618) % 1, 0.65, 0.95)
            sim.setShapeColor(body, "", sim.colorcomponent_ambient_diffuse, list(rgb))


def picture(camera, title):
    """The camera image (rendered during the last step) as BGR, with a title."""
    img, (w, h) = sim.getVisionSensorImg(camera)
    img = cv2.cvtColor(np.flipud(np.frombuffer(img, np.uint8).reshape(h, w, 3)), cv2.COLOR_RGB2BGR)
    cv2.putText(img, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    return img


@contextmanager
def physics_engine():
    """Select the requested engine and restore scene settings afterwards."""
    old_engine = sim.getInt32Param(sim.intparam_dynamic_engine)
    old_realtime = sim.getBoolProperty(sim.handle_scene, "realtimeSimulation")
    old_cone = sim.getProperty(sim.handle_scene, "mujoco.cone") if ENGINE == "mujoco" else None
    try:
        sim.setBoolProperty(sim.handle_scene, "realtimeSimulation", False)
        sim.setInt32Param(sim.intparam_dynamic_engine, getattr(sim, f"physics_{ENGINE}"))
        if ENGINE == "mujoco":
            sim.setProperty(sim.handle_scene, "mujoco.cone", 1)
        print(f"Physics engine: {ENGINE}", flush=True)
        yield
    finally:
        sim.setBoolProperty(sim.handle_scene, "realtimeSimulation", old_realtime)
        sim.setInt32Param(sim.intparam_dynamic_engine, old_engine)
        if ENGINE == "mujoco":
            sim.setProperty(sim.handle_scene, "mujoco.cone", old_cone)


if "load" in FLAGS:
    root, camera, bodies = load_scene()
else:
    t_build = time.time()
    root, camera = build_table()
    bodies = spawn_all(root)
    print(f"built in {time.time() - t_build:.1f} s")

# Apply to newly built and loaded scenes alike, including the table and raw bodies.
if ENGINE == "bullet":
    for shape in sim.getObjectsInTree(root, sim.object_shape_type, 0):
        sim.setFloatProperty(shape, "bullet.friction", BULLET_FRICTION)
    print(f"Bullet friction: {BULLET_FRICTION} on all shapes (table and objects)", flush=True)
if "save" in FLAGS:
    save_scene(root)

show_proxies(bodies, "show" in FLAGS)
with physics_engine(), stepped_simulation(sim):
    dt = sim.getSimulationTimeStep()
    for _ in range(int(3.0 / dt)):  # fall and settle
        sim.step()
    if "show" in FLAGS and PROXY != "raw":
        show_proxies(bodies, False)
        sim.step()  # refresh the scan comparison image
    scans = picture(camera, "what you see: the scans")
    show_proxies(bodies, True)
    sim.step()  # sensors render during a step
    proxies = picture(camera, (f"raw mesh ({ENGINE} collision: convex hull)" if ENGINE in ("mujoco", "newton")
                       else "raw mesh (Bullet)") if PROXY == "raw"
                      else f"what the physics engine sees: {PROXIES[PROXY]}")
    show_proxies(bodies, "show" in FLAGS)  # keep collision pieces visible after the reveal

    at_3s = {n: com(h) for n, h in bodies.items()}
    for _ in range(int(2.0 / dt)):  # should now be at rest
        sim.step()
    drift = {n: np.linalg.norm(com(h) - at_3s[n]) * 1000 for n, h in bodies.items()}
    worst = max(drift, key=drift.get)
    print(f"\nat rest? drift between 3 s and 5 s: median {np.median(list(drift.values())):.1f} mm, "
          f"max {drift[worst]:.1f} mm ({worst})")
    for name in INTO_BOWL:
        x, y, z = com(bodies[name])
        r = math.hypot(x, y)
        where = "rolled off" if r > 0.07 else "in the bowl" if z < TABLE_TOP + 0.07 else "ON TOP of the bowl"
        print(f"  {name} with the bowl as {PROXIES[PROXY]}: {where}  (r = {r * 100:4.1f} cm, z = {z * 100:4.1f} cm)")

os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(os.path.join(OUT_DIR, f"ycb_{PROXY or 'auto'}.png"), np.hstack([scans, proxies]))
print(f"saved out/ycb_{PROXY or 'auto'}.png (scans | proxies)")
if PROXY == "raw":
    print("the original textured meshes are visible; no collision proxies were generated")
elif "show" in FLAGS:
    print("the proxies are visible in the scene, the scans are in layer 9")
print("stopping restores the start poses: press Play in CoppeliaSim to watch them fall again")
