"""Manual numerical IK: edit target pose, then click once per iteration.
Run: python scripts/lecture1/jacobian_ik_step.py (keep simulation STOPPED).
The UI, KUKA setup and drawing code live in support/jacobian_ik_ui.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import time
import numpy as np
from lecture1.jacobian_demo import geometric_jacobian
from lecture1.support.jacobian_ik_ui import IKPanel

GAIN = .6
PINV_RCOND = 1e-6  # Treat tiny model/roundoff singular values as zero.
MAX_STEP = math.radians(5)


def rotation_vector(R):
    """SO(3) logarithm, including small-angle and near-pi cases; radians."""
    angle = math.acos(np.clip((np.trace(R)-1)/2, -1, 1))
    skew = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    if angle < 1e-7:
        return .5*skew
    if math.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((R+R.T)/2)
        axis = vectors[:, -1]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return angle*axis
    return angle/(2*math.sin(angle))*skew


def pose_error(current, target):
    # Both errors are WORLD-frame quantities, matching the geometric Jacobian.
    return np.r_[target[:3,3]-current[:3,3],
                 rotation_vector(target[:3,:3] @ current[:3,:3].T)]


def correction(J, error):
    # Moore-Penrose pseudoinverse: dq = gain * J^+ * error.
    dq = GAIN * np.linalg.pinv(J, rcond=PINV_RCOND) @ error
    return dq * min(1., MAX_STEP/max(np.max(np.abs(dq)), 1e-12))


def iterate(ui):
    """Exactly one update: read pose -> error -> Jacobian -> dq -> apply q+dq."""
    started = time.perf_counter()
    sim, joints = ui.sim, ui.joints
    current, target = ui.pose(ui.eef), ui.pose(ui.target)
    error = pose_error(current, target)
    q = np.array([sim.getJointPosition(joint) for joint in joints])
    J = geometric_jacobian(sim, joints, ui.eef)
    solve_started = time.perf_counter()
    dq = correction(J, error)
    solve_ms = (time.perf_counter()-solve_started)*1000
    q_next = np.clip(q+dq, ui.lower, ui.upper)
    for joint, value in zip(joints, q_next):
        sim.setJointPosition(joint, float(value))  # Numerical update, no motor wait.
    after = pose_error(ui.pose(ui.eef), target)
    elapsed_ms = (time.perf_counter()-started)*1000
    ui.report_iteration(current, error, q_next-q, after, solve_ms, elapsed_ms)
    return error, after


def main():
    try:
        with IKPanel(pose_error, pinv_rcond=PINV_RCOND) as ui:
            while True:
                action = ui.poll()
                if action == 'exit':
                    break
                if action == 'iterate':
                    iterate(ui)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
