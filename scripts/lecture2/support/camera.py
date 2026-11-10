"""RGB-D Camera, point-cloud geometry and live grasp/cloud display."""

import math
import os

import numpy as np

from shared.common import CAM_FAR, CAM_FOV, CAM_NEAR, CUBE_SIZE, OUT_DIR, first, pose


def intrinsics(sim, cam, fov=None):
    """(f, cx, cy, w, h) in pixels. The view angle applies to the LARGER image side."""
    w = sim.getObjectInt32Param(cam, sim.visionintparam_resolution_x)
    h = sim.getObjectInt32Param(cam, sim.visionintparam_resolution_y)
    if fov is None:
        fov = sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
    f = (max(w, h) / 2) / math.tan(fov / 2)
    return f, w / 2, h / 2, w, h


def depth_to_points(depth, K, T_cam, stride=1, max_depth=None, rgb=None):
    """Every stride-th pixel with a hit -> world point. Pixels at max_depth (no hit) are dropped.
    Returns points (N, 3), plus colours (N, 3) in 0..1 if rgb is given."""
    f, cx, cy = K[:3]
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    Z = depth[v, u]
    keep = np.isfinite(Z) & (Z > 0) & (Z < (max_depth if max_depth is not None else np.inf) * 0.999)
    u, v, Z = u[keep], v[keep], Z[keep]
    p_cam = np.stack([(cx - (u + 0.5)) * Z / f, (cy - (v + 0.5)) * Z / f, Z, np.ones_like(Z)])
    points = (np.asarray(T_cam) @ p_cam)[:3].T
    if rgb is None:
        return points
    return points, rgb[v, u] / 255.0


def project(points, K, T_cam):
    """World points -> pixel (u: column, v: row from the top). Inverse of depth_to_points."""
    f, cx, cy = K[:3]
    p = (np.linalg.inv(T_cam) @ np.c_[points, np.ones(len(points))].T)[:3]
    return np.stack([cx - p[0] * f / p[2] - 0.5, cy - p[1] * f / p[2] - 0.5], axis=1)


def depth_segments(depth, mask, max_step=0.006, min_pixels=40):
    """Region growing on the depth image: neighbouring pixels belong together when their depths
    differ by less than max_step. Object borders are depth jumps, so objects that touch in 3D
    (one Euclidean cluster) still split. Returns a list of boolean pixel masks, largest first."""
    import cv2

    jump = np.zeros(depth.shape, bool)
    jump[:, 1:] |= np.abs(np.diff(depth, axis=1)) > max_step
    jump[1:, :] |= np.abs(np.diff(depth, axis=0)) > max_step
    n, labels = cv2.connectedComponents((mask & ~jump).astype(np.uint8), connectivity=4)
    segs = [labels == k for k in range(1, n)]
    return sorted((m for m in segs if m.sum() >= min_pixels), key=lambda m: m.sum(), reverse=True)


def pixel_points(depth, K, T_cam):
    """World point of EVERY pixel, as an (H, W, 3) image (no pixel dropped, unlike depth_to_points)."""
    f, cx, cy = K[:3]
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    p_cam = np.stack([(cx - (u + 0.5)) * depth / f, (cy - (v + 0.5)) * depth / f, depth, np.ones_like(depth)], -1)
    return (p_cam @ np.asarray(T_cam).T)[..., :3]


def principal_axes(points):
    """PCA: centre, axes (columns, largest spread first), extents (max - min along each axis)."""
    c = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    local = (points - c) @ Vt.T
    return c, Vt.T, local.max(axis=0) - local.min(axis=0)


def save_ply(path, points, colors=None):
    """ASCII PLY: open it in MeshLab, CloudCompare or Open3D."""
    with open(path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
        )
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            np.savetxt(f, points, fmt="%.5f")
        else:
            np.savetxt(f, np.c_[points, (np.asarray(colors) * 255).astype(int)], fmt="%.5f %.5f %.5f %d %d %d")


def read_rgbd(sim, camera):
    """Read an already-handled sensor: RGB uint8, optical-axis depth float32 [m].

    Call sim.handleVisionSensor(camera) first for explicit-handling sensors.
    Otherwise sim.step() renders automatically. Far-plane pixels are no-return values.
    """
    data, (w, h) = sim.getVisionSensorImg(camera)
    rgb = np.flipud(np.frombuffer(data, np.uint8).reshape(h, w, 3)).copy()
    result = sim.getVisionSensorDepth(camera, 1)  # bit 0: metres, not normalised depth
    data = result[0] if isinstance(result, (tuple, list)) else result
    depth = np.flipud(np.frombuffer(data, np.float32).reshape(h, w)).copy()
    return rgb, depth


def grab_images(sim, cam, near=CAM_NEAR, far=CAM_FAR):
    """RGB (H,W,3 uint8) and metric depth (H,W float, distance along the optical axis)."""
    img, res = sim.getVisionSensorImg(cam)
    w, hgt = res
    rgb = np.frombuffer(img, dtype=np.uint8).reshape(hgt, w, 3)
    rgb = np.flipud(rgb).copy()  # CoppeliaSim images are stored bottom row first
    buf = first(sim.getVisionSensorDepth(cam))  # normalised 0..1 between near and far
    depth = np.frombuffer(buf, dtype=np.float32).reshape(hgt, w)
    depth = near + np.flipud(depth) * (far - near)
    return rgb, depth


def save_images(rgb, depth, prefix):
    import cv2

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    d = (255 * (depth - depth.min()) / max(1e-6, np.ptp(depth))).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_depth.png"), cv2.applyColorMap(d, cv2.COLORMAP_VIRIDIS))


def detect_red(rgb):
    """Pixel centroid (u: column, v: row from top) of red pixels, and the mask."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    mask = (r > 150) & (g < 80) & (b < 80)
    if mask.sum() < 20:
        return None, mask
    vs, us = np.nonzero(mask)
    return (us.mean(), vs.mean()), mask


def pixel_to_world(sim, cam, uv, Z, fov=CAM_FOV):
    """Pinhole back-projection.  CoppeliaSim camera frame: looks along +z, image up = +y,
    so image right = -x.  The view angle applies to the larger image side."""
    w = sim.getObjectInt32Param(cam, sim.visionintparam_resolution_x)
    hgt = sim.getObjectInt32Param(cam, sim.visionintparam_resolution_y)
    f = (max(w, hgt) / 2) / math.tan(fov / 2)
    u, v = uv
    p_cam = np.array([(w / 2 - (u + 0.5)) * Z / f, (hgt / 2 - (v + 0.5)) * Z / f, Z, 1.0])
    return (pose(sim, cam) @ p_cam)[:3]


def locate_cube(sim, h, save_prefix=None):
    """Return the cube centre in world coordinates, estimated from RGB-D only."""
    rgb, depth = grab_images(sim, h["camera"])
    if save_prefix:
        save_images(rgb, depth, save_prefix)
    uv, mask = detect_red(rgb)
    if uv is None:
        return None
    Z_top = float(np.median(depth[mask]))  # depth of the top face
    p_top = pixel_to_world(sim, h["camera"], uv, Z_top)
    return p_top - np.array([0, 0, CUBE_SIZE / 2])


def grasp_segments(grasp, width, height=0.005):
    """GraspNet-style wire gripper: two fingers, rear bar and approach stem.

    This is still a top-down pose: the approach direction is fixed, not predicted.
    """
    centre = np.asarray(grasp["top"], float) + [0, 0, height]
    jaw = np.r_[grasp["jaw"], 0.0]
    jaw /= np.linalg.norm(jaw)
    side = np.cross([0, 0, 1], jaw) * 0.003
    rear = np.array([0.0, 0.0, 0.04])
    a, b = centre - width / 2 * jaw, centre + width / 2 * jaw
    segments = []
    for tip in (a, b):
        corners = [tip - side, tip + side, tip + side + rear, tip - side + rear]
        segments.extend(np.r_[corners[i], corners[(i + 1) % 4]] for i in range(4))
    segments.extend((np.r_[a + rear, b + rear], np.r_[centre + rear, centre + rear + [0, 0, 0.025]]))
    return np.array(segments)


class LivePointCloud:
    def __init__(self, cell, period=0.1):
        self.cell, self.sim, self.period = cell, cell.sim, period
        self.camera = cell.h["binCamera"]
        self.K = intrinsics(self.sim, self.camera)
        self.last_time, self.updates, self.count = -1.0, 0, 0
        self.markers = None  # create after simulation starts: startup clears old drawings
        self.predictor, self.selected = None, None
        self.last_prediction, self.prediction_updates = -1.0, 0
        self.points, self.colors = np.empty((0, 3)), np.empty((0, 3))
        self.display_grasps = []
        old = self.sim.getObject("/Live_RGBD_Cloud", {"noError": True})
        if old != -1:
            self.sim.removeObjects([old])
        self.cloud = self.sim.createPointCloud(0.02, 100, 8 | 16, 3.0)
        self.sim.setObjectAlias(self.cloud, "Live_RGBD_Cloud")
        self.sim.setObjectSpecialProperty(self.cloud, 0)  # never seen by sensors or collision checks
        self.sim.setObjectPosition(self.cloud, [0, 0.65, 0], self.sim.handle_world)

    def predictions(self, raw, accepted, selected=None):
        """Score-coloured 3D grippers; accepted proposals use depth-refined widths."""
        sim = self.sim
        self.selected = selected
        if self.markers is None:
            flags = sim.drawing_lines | sim.drawing_local | sim.drawing_overlay
            # Draw the selection halo before the coloured lines.
            self.selection = sim.addDrawingObject(flags, 7, 0, self.cloud, 32, [1.0, 1.0, 1.0])
            self.markers = [
                sim.addDrawingObject(flags, 3, 0, self.cloud, 256, [0.6 + 0.4 * i / 7, 0.0, 0.6 * (1 - i / 7)])
                for i in range(8)
            ]
            print(
                "Point-cloud grasps: purple -> red = low -> high relative network score; "
                "white outline = selected. Fixed top-down approach, not 6-DoF prediction.",
                flush=True,
            )
        proposals = list((accepted if accepted else raw)[:8])
        if selected is not None and not any(g is selected for g in proposals):
            proposals = proposals[:7] + [selected]
        self.display_grasps = proposals
        for handle in [self.selection, *self.markers]:
            sim.addDrawingObjectItem(handle, None)
        if not proposals:
            return
        scores = [g["quality"] for g in proposals]
        low, high = min(scores), max(scores)
        bins = [[] for _ in self.markers]
        for g in proposals:
            fraction = (g["quality"] - low) / (high - low) if high - low > 1e-8 else 0.5
            width = g.get("opening", g.get("network_opening"))
            lines = grasp_segments(g, width)
            bins[round(7 * fraction)].append(lines)
            if g is selected:
                sim.addDrawingObjectItem(self.selection | sim.handleflag_addmultiple, lines.ravel().tolist())
        for handle, items in zip(self.markers, bins):
            if items:
                sim.addDrawingObjectItem(handle | sim.handleflag_addmultiple, np.array(items).ravel().tolist())

    def update(self):
        t = self.sim.getSimulationTime()
        if t - self.last_time < self.period - 1e-7:
            return
        rgb, depth = read_rgbd(self.sim, self.camera)
        points, colors = depth_to_points(depth, self.K, pose(self.sim, self.camera), stride=2, max_depth=1.2, rgb=rgb)
        self.points, self.colors = points, colors
        self.sim.removePointsFromPointCloud(self.cloud, 0, None, 0.0)
        # Local coordinates equal unshifted world coordinates; the object provides display offset.
        self.sim.insertPointsIntoPointCloud(
            self.cloud, 3, points.ravel().tolist(), np.rint(colors * 255).astype(np.uint8).ravel().tolist()
        )
        # Never move the user's viewport when cloud data arrives.
        if self.predictor is not None and t - self.last_prediction >= 0.5 - 1e-7:
            _, accepted = self.predictor(rgb, depth, pose(self.sim, self.camera))
            self.predictions(self.cell.network_predictions, accepted, self.selected)
            self.last_prediction = t
            self.prediction_updates += 1
        self.last_time, self.count = t, len(points)
        self.updates += 1

    def __enter__(self):
        self.cell.sim = _ObservedSimulation(self.sim, self._step)
        return self

    def _step(self):
        self.update()

    def __exit__(self, *args):
        self.cell.sim = self.sim
        print(f"Live point cloud: {self.updates} updates, {self.count:,} points in final frame.", flush=True)


class _ObservedSimulation:
    """Forward the API; observe steps in all move/close/wait helpers, not just detection."""

    def __init__(self, sim, after_step):
        self._sim, self._after_step = sim, after_step

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def step(self):
        result = self._sim.step()
        self._after_step()
        return result


class Camera:
    """A bound vision-sensor handle; returned depth is along optical +Z, in metres."""

    def __init__(self, sim, handle):
        self.sim, self.handle = sim, handle

    def rgbd(self):
        return read_rgbd(self.sim, self.handle)

    def pose(self):
        return pose(self.sim, self.handle)

    def intrinsics(self):
        return intrinsics(self.sim, self.handle)

    def points(self, depth, rgb=None, stride=1, max_depth=None):
        return depth_to_points(depth, self.intrinsics(), self.pose(), stride=stride, max_depth=max_depth, rgb=rgb)
