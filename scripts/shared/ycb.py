"""YCB object set (real scanned household objects): download, load, spawn into CoppeliaSim.

Data: the "google_16k" scans (16k triangles + 4096^2 texture) from the YCB benchmark,
https://www.ycbbenchmarks.com, licence CC BY 4.0. Calli et al., "Benchmarking in Manipulation
Research: The YCB Object and Model Set", 2015 (masses below: Table II of that paper).

Only the objects you ask for are downloaded (~11 MB each), into scripts/data/ycb/<name>/.
CoppeliaSim may run on another machine than Python (here: the host vs. the container), so we
do not let it import files: we parse the mesh in Python and send vertices + texture with
sim.createShape.
"""
import io
import os
import shutil
import tarfile
import urllib.request

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ycb")
URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz"
FILES = ("textured.obj", "texture_map.png")

# mass in grams, Table II of Calli et al. 2015 (the paper numbers objects differently)
MASS_G = {
    "001_chips_can": 205, "002_master_chef_can": 414, "003_cracker_box": 453, "004_sugar_box": 514,
    "005_tomato_soup_can": 349, "006_mustard_bottle": 431, "007_tuna_fish_can": 171, "008_pudding_box": 187,
    "009_gelatin_box": 97, "010_potted_meat_can": 370, "011_banana": 66, "012_strawberry": 18, "013_apple": 68,
    "014_lemon": 29, "015_peach": 33, "016_pear": 49, "017_orange": 47, "018_plum": 25, "019_pitcher_base": 178,
    "022_windex_bottle": 1022, "023_wine_glass": 133, "024_bowl": 147, "025_mug": 118, "029_plate": 279,
    "035_power_drill": 895, "037_scissors": 82, "048_hammer": 688, "052_extra_large_clamp": 202,
    "054_softball": 191, "056_tennis_ball": 58, "057_racquetball": 41, "058_golf_ball": 46, "061_foam_brick": 28,
}
# convexity = mesh volume / convex hull volume. Below this the hull changes the object too much
# (a bowl's hull is a closed dome, a banana's hull rocks on its curved belly and never stops):
# use a convex decomposition instead. YCB: bowl 0.04, mug 0.26, banana 0.65, cans/bottles ~0.9, fruit ~1.
MIN_CONVEXITY = 0.8
VISIBLE_LAYER, HIDDEN_LAYER = 1 << 0, 1 << 8  # layers 1 and 9: only layers 1-8 are shown by default
# simConvex.vhacd defaults to concavity 0.1, 40x coarser than V-HACD's own default 0.0025: the mug's
# pieces then fill 70 % of its inside (35 % with 0.0025), the bowl's 49 % (14 %), the banana's 37 % (27 %)
VHACD_PARAMS = {"concavity": 0.0025}
# CoACD (Wei et al. 2022, pip install coacd): cuts along any plane, measures concavity on the surface
# rather than on voxels; fills mug / bowl / banana 15 / 8 / 9 % instead of 35 / 14 / 27 %, but slower
COACD_PARAMS = {"threshold": 0.05}


def download(name):
    """Cached download of one object; returns its folder. Only the mesh and texture are kept."""
    folder = os.path.join(DATA_DIR, name)
    if all(os.path.exists(os.path.join(folder, f)) for f in FILES):
        return folder
    print(f"  downloading {name} ...", end="", flush=True)
    with urllib.request.urlopen(URL.format(name=name), timeout=120) as r:
        data = r.read()
    tmp = folder + ".part"  # a fresh, empty folder for the untrusted archive
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        wanted = [m for m in tar.getmembers() if os.path.basename(m.name) in FILES and m.isfile()]
        tar.extractall(tmp, members=wanted, filter="data")  # "data" filter: no absolute paths, no links out
    os.makedirs(folder, exist_ok=True)
    for m in wanted:
        shutil.move(os.path.join(tmp, m.name), os.path.join(folder, os.path.basename(m.name)))
    shutil.rmtree(tmp)
    print(f" {len(data) / 1e6:.1f} MB")
    return folder


def load(name, texture_size=1024):
    """Vertices (n,3) [m], triangles (m,3), texture coordinates per triangle corner (m,3,2),
    and the texture as an RGB image (texture_size^2)."""
    folder = download(name)
    V, UV, F, FT = [], [], [], []
    with open(os.path.join(folder, "textured.obj")) as f:
        for line in f:
            tag, *vals = line.split()
            if tag == "v":
                V.append([float(x) for x in vals[:3]])
            elif tag == "vt":
                UV.append([float(x) for x in vals[:2]])
            elif tag == "f":  # "v/vt/vn" per corner, 1-based; the scans are triangles
                corners = [c.split("/") for c in vals[:3]]
                F.append([int(c[0]) - 1 for c in corners]) #vertex indices
                FT.append([int(c[1]) - 1 for c in corners]) #UV indices
    tex = cv2.imread(os.path.join(folder, "texture_map.png"), cv2.IMREAD_COLOR)
    tex = cv2.cvtColor(cv2.resize(tex, (texture_size, texture_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.array(V), np.array(F), np.array(UV)[np.array(FT)], tex


def create_textured_shape(sim, V, F, UVc, tex):
    """A (static, non-respondable) textured mesh shape, created at the mesh coordinates
    (the scan's own frame = the world frame, until we move it). createShape wants the texture
    coordinates per index (triangle corner) and the texture as raw RGB bytes."""
    options = 64  # bit 6: flip the texture vertically (image rows top-down, OBJ v axis bottom-up)
    return sim.createShape(options, 0.5, V.ravel().tolist(), F.ravel().tolist(), None,
                           UVc.reshape(-1).tolist(), tex.tobytes(), [tex.shape[1], tex.shape[0]])


def vhacd_pieces(sim, simConvex, name, visual):
    """Convex decomposition of the mesh as [(vertices, triangles)], in mesh coordinates.
    Cached next to the mesh, under a name that contains the parameters."""
    tag = "_".join(f"{k}{v}" for k, v in sorted(VHACD_PARAMS.items()))
    cache = os.path.join(DATA_DIR, name, f"vhacd_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return [(d[f"v{k}"], d[f"f{k}"]) for k in range(len(d.files) // 2)]
    print(f"  convex decomposition of {name} (once, then cached) ...", flush=True)
    pieces = []
    for h in simConvex.vhacd(visual, VHACD_PARAMS):
        v, f, _ = sim.getShapeMesh(h)  # vertices in the piece's own frame
        T = np.vstack([np.array(sim.getObjectMatrix(h, sim.handle_world)).reshape(3, 4), [0, 0, 0, 1]])
        v = (T @ np.c_[np.reshape(v, (-1, 3)), np.ones(len(v) // 3)].T).T[:, :3]  # -> mesh coordinates
        pieces.append((v, np.reshape(f, (-1, 3))))
        sim.removeObjects([h])
    np.savez(cache, **{f"v{k}": v for k, (v, _) in enumerate(pieces)}, **{f"f{k}": f for k, (_, f) in enumerate(pieces)})
    return pieces


# ================================================================ sphere decomposition
def surface_samples(V, F, n, rng):
    """n points spread uniformly over the surface, with the outward normal of their triangle."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(area[:, None], 1e-12)
    if np.einsum("ij,ij->i", a, cross).sum() < 0:  # negative signed volume: triangles wound inwards
        normal = -normal
    tri = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[tri] + u[:, None] * (b - a)[tri] + v[:, None] * (c - a)[tri], normal[tri]


def nearest(Q, P, chunk=2048):
    """Distance from each point of Q to its nearest point of P, and that point's index."""
    dist, idx = np.empty(len(Q)), np.empty(len(Q), int)
    for i in range(0, len(Q), chunk):
        d2 = ((Q[i:i + chunk, None, :] - P[None, :, :]) ** 2).sum(-1)
        idx[i:i + chunk] = d2.argmin(1)
        dist[i:i + chunk] = np.sqrt(d2.min(1))
    return dist, idx


def inside_mesh(Q, V, F, chunk=256):
    """Ray parity: a ray going up from q crosses a closed surface an odd number of times iff q is
    inside. Exact for any wall thickness (unlike 'which side of the nearest surface point am I')."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(det) > 1e-15  # triangles seen edge-on from above are never crossed
    a, b, c, det = a[ok], b[ok], c[ok], det[ok]
    lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    Q = Q + np.array([1.3e-7, 0.7e-7, 0])  # off any shared edge or vertex
    out = np.zeros(len(Q), bool)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk, None, :]
        hit = (q[..., 0] >= lo[:, 0]) & (q[..., 0] <= hi[:, 0]) & (q[..., 1] >= lo[:, 1]) & (q[..., 1] <= hi[:, 1])
        u = ((q[..., 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[..., 1] - a[:, 1])) / det
        v = ((b[:, 0] - a[:, 0]) * (q[..., 1] - a[:, 1]) - (q[..., 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        hit &= (u >= 0) & (v >= 0) & (u + v <= 1) & (z > q[..., 2])
        out[i:i + chunk] = hit.sum(1) % 2 == 1
    return out


def fit_spheres(V, F, max_spheres=256, coverage=0.99, tol=0.004, r_min=0.004, n_samples=3000, seed=0):
    """Approximate a closed mesh by spheres. Returns centres (k,3), radii (k,), covered fraction.
    1. candidates: step from surface points inwards (along -normal) by 0.5, 1, 2, 4 ... mm; keep
       those inside the mesh; the largest sphere that fits = distance to the nearest surface point
       (walls thinner than 2 r_min get spheres of radius r_min: a bit thicker, but closed)
    2. greedy cover: take the sphere that reaches (within tol) most still-uncovered surface points"""
    rng = np.random.default_rng(seed)
    P, N = surface_samples(V, F, n_samples, rng)
    size = (V.max(0) - V.min(0)).max()
    depths = 0.0005 * 2 ** np.arange(int(np.log2(size / 2 / 0.0005)) + 1)
    C = np.concatenate([P - N * t for t in depths])
    R, near = nearest(C, P)
    keep = np.einsum("ij,ij->i", C - P[near], N[near]) < 0  # cheap pre-filter: behind the nearest surface
    keep[keep] = inside_mesh(C[keep], V, F)                  # exact
    C, R = C[keep], np.maximum(R[keep], r_min)
    reach = np.zeros((len(C), len(P)), bool)
    for i in range(0, len(C), 1024):
        reach[i:i + 1024] = ((C[i:i + 1024, None] - P[None]) ** 2).sum(-1) <= (R[i:i + 1024, None] + tol) ** 2
    covered, chosen = np.zeros(len(P), bool), []
    while covered.mean() < coverage and len(chosen) < max_spheres:
        gain = (reach & ~covered).sum(1)
        k = int((gain + 1e-3 * R).argmax())  # ties: the bigger sphere
        if gain[k] == 0:
            break
        chosen.append(k)
        covered |= reach[k]
    return C[chosen], R[chosen], covered.mean()


def sphere_set(name, V, F):
    """fit_spheres, cached next to the mesh (a few seconds per object)."""
    cache = os.path.join(DATA_DIR, name, "spheres.npz")
    if not os.path.exists(cache):
        print(f"  sphere decomposition of {name} (once, then cached) ...", flush=True)
        C, R, cov = fit_spheres(V, F)
        np.savez(cache, centres=C, radii=R, coverage=cov)
    d = np.load(cache)
    return d["centres"], d["radii"], float(d["coverage"])


def coacd_pieces(name, V, F):
    """Convex decomposition by CoACD as [(vertices, triangles)], in mesh coordinates, cached.
    Flat pieces (no volume, CoACD sometimes makes them) are dropped."""
    tag = "_".join(f"{k}{v}" for k, v in sorted(COACD_PARAMS.items()))
    cache = os.path.join(DATA_DIR, name, f"coacd_{tag}.npz")
    if not os.path.exists(cache):
        import coacd  # optional dependency: only this proxy mode needs it
        coacd.set_log_level("error")
        print(f"  CoACD decomposition of {name} (once, then cached; can take a minute) ...", flush=True)
        parts = [(np.asarray(v, float), np.asarray(f, int)) for v, f in coacd.run_coacd(coacd.Mesh(V, F), **COACD_PARAMS)]
        parts = [(v, f) for v, f in parts if volume(v, f) > 1e-9]
        np.savez(cache, **{f"v{k}": v for k, (v, _) in enumerate(parts)}, **{f"f{k}": f for k, (_, f) in enumerate(parts)})
    d = np.load(cache)
    return [(d[f"v{k}"], d[f"f{k}"]) for k in range(len(d.files) // 2)]


def volume(V, F):
    """Volume of a closed triangle mesh: the signed tetrahedra (origin, triangle) add up."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6


def _build(sim, simConvex, simOpenMesh, name, proxy, V, F, UVc, tex):
    visual = create_textured_shape(sim, V, F, UVc, tex)
    sim.setBoolProperty(visual, "dynamic", False)      # only drawn: say so explicitly
    sim.setBoolProperty(visual, "respondable", False)
    # hull of a decimated copy (80 % of the triangles removed): the raw hull of a scan can have
    # >10k triangles; removing 95 % made e.g. the coffee can a crude prism that never stops wobbling
    coarse = simOpenMesh.decimate(visual, {"percentage": 0.8})
    body = simConvex.hull([coarse])
    sim.removeObjects([coarse])
    hv, hf, _ = sim.getShapeMesh(body)  # a few hundred triangles: quick to read back
    spawn.convexity = volume(V, F) / volume(np.reshape(hv, (-1, 3)), np.reshape(hf, (-1, 3)))
    if proxy is None:  # only automatic mode may substitute a hull
        proxy = "hull" if spawn.convexity >= MIN_CONVEXITY else "vhacd"
    # An explicit CoACD request must run CoACD even for nearly convex scans.
    # Otherwise scanned boxes silently keep the noisy hull contact surface.
    spawn.detail = f"hull, {len(hf) // 3} triangles"
    if proxy in ("vhacd", "coacd"):
        sim.removeObjects([body])
        parts = vhacd_pieces(sim, simConvex, name, visual) if proxy == "vhacd" else coacd_pieces(name, V, F)
        pieces = [sim.createShape(0, 0.5, v.ravel().tolist(), f.ravel().tolist()) for v, f in parts]
        body = sim.groupShapes(pieces, False) if len(pieces) > 1 else pieces[0]
        spawn.detail = f"{len(pieces)} {proxy.replace('vhacd', 'V-HACD').replace('coacd', 'CoACD')} pieces"
    elif proxy == "spheres":
        sim.removeObjects([body])
        C, R, cov = sphere_set(name, V, F)
        pieces = []
        for c, r in zip(C, R):
            h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * float(r)] * 3, 0)
            sim.setObjectPosition(h, [float(x) for x in c], sim.handle_world)
            pieces.append(h)
        body = sim.groupShapes(pieces, False) if len(pieces) > 1 else pieces[0]  # False: stays primitive
        spawn.detail = f"{len(pieces)} spheres, {cov * 100:.0f} % covered"
    return visual, body


def spawn(sim, simConvex, simOpenMesh, name, parent, proxy=None, xy=(0.0, 0.0), z_bottom=0.0):
    """One YCB object as in lecture1/shape_types.py: a convex dynamic proxy with the real mass
    (hidden in layer 9) + the scanned, textured mesh as a visual-only child.
    proxy: "hull", "vhacd" (convex decomposition), "spheres" (sphere decomposition: a primitive
    compound), None = hull or vhacd chosen by convexity (MIN_CONVEXITY), "coacd" = CoACD for every object (no convexity-based hull fallback).
    Returns the proxy. For the last object, spawn.convexity is its mesh / hull volume ratio and
    spawn.detail describes the proxy (e.g. "63 V-HACD pieces", "210 spheres, 99 % covered").
    The object is placed with its bounding-box centre above xy and its lowest point at z_bottom,
    under `parent` (e.g. /Lab, so that clear_scene removes it: new objects start at the scene root)."""
    V, F, UVc, tex = load(name)
    before = set(sim.getObjectsInTree(sim.handle_scene))
    try:
        visual, body = _build(sim, simConvex, simOpenMesh, name, proxy, V, F, UVc, tex)
        sim.computeMassAndInertia(body, 1000.0)                  # inertia from the shape ...
        sim.setFloatProperty(body, "mass", MASS_G[name] / 1000)  # ... scaled to the real mass
        sim.setBoolProperty(body, "dynamic", True)
        sim.setBoolProperty(body, "respondable", True)
        sim.setIntProperty(body, "layer", HIDDEN_LAYER)  # the proxy is not drawn, the scan is
        # MuJoCo: a round scan rolls for ever without rolling friction, which needs condim 6
        # (3 = sliding friction only). Ignored by the other engines.
        sim.setFloatArrayProperty(body, "mujoco.friction", [1.0, 0.01, 0.01])  # sliding, torsional, rolling
        sim.setIntProperty(body, "mujoco.condim", 6)
        sim.setObjectParent(visual, body, True)
        sim.setObjectAlias(body, name)
        sim.setObjectAlias(visual, name + "_visual")
        lo, hi = V.min(axis=0), V.max(axis=0)  # the mesh is still at its own coordinates
        shift = [xy[0] - (lo[0] + hi[0]) / 2, xy[1] - (lo[1] + hi[1]) / 2, z_bottom - lo[2]]
        sim.setObjectPosition(body, list(np.add(sim.getObjectPosition(body, sim.handle_world), shift)), sim.handle_world)
        sim.setObjectParent(body, parent, True)
    except BaseException:  # leave no half-built object at the scene root (clear_scene would miss it)
        sim.removeObjects([h for h in sim.getObjectsInTree(sim.handle_scene) if h not in before])
        raise
    return body
