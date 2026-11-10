"""KUKA forward kinematics using the lecture's DH convention.
Run: python scripts/lecture1/dh_forward_kinematics.py
Angles q are in radians; DH twists alpha are listed in degrees; lengths are metres.
Scene setup, frame drawings and model comparison: support/dh_demo.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np
from lecture1.support.dh_demo import DHScene

# DH table: each row contains (a[i-1], alpha[i-1], d[i]).
DH_TABLE = [
    (0,   0, .360177),
    (0,  90, 0),
    (0, -90, .420000),
    (0,  90, 0),
    (0, -90, .400000),
    (0,  90, 0),
    (0, -90, .110999),
]
JOINT_SIGNS = np.array([1, -1, 1, 1, 1, -1, 1])  # Model joint axes -> DH joint axes.
AMPLITUDES = np.radians([40, 35, 35, -40, 35, 35, 40])
DURATION = 20.0
MARKER_INTERVAL = .05


def dh_transform(a, alpha_deg, d, theta):
    """One link transform: A = Tx(a) @ Rx(alpha) @ Rz(theta) @ Tz(d)."""
    alpha = math.radians(alpha_deg)
    c, s = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [c,    -s,     0,      a],
        [s*ca,  c*ca, -sa, -d*sa],
        [s*sa,  c*sa,  ca,  d*ca],
        [0,      0,    0,      1],
    ])


def forward_kinematics(q):
    """Multiply A1 ... A7; return every frame T_0_0 ... T_0_7 for drawing."""
    frames = [np.eye(4)]
    for (a, alpha, d), theta in zip(DH_TABLE, q):
        A = dh_transform(a, alpha, d, theta)
        frames.append(frames[-1] @ A)
    return frames  # frames[-1] is the EEF pose in the DH base frame.


def main():
    with DHScene(marker_interval=MARKER_INTERVAL) as demo:
        sim = demo.sim
        while sim.getSimulationTime() < DURATION:
            # 1. Command smooth motion, then advance one simulation step.
            t = sim.getSimulationTime()
            targets = AMPLITUDES * math.sin(t)
            for joint, target in zip(demo.joints, targets):
                sim.setJointTargetPosition(joint, float(target))
            sim.step()

            # 2. Read actual angles and convert to the DH axis convention.
            q_sim = np.array([sim.getJointPosition(j) for j in demo.joints])
            q_dh = JOINT_SIGNS * q_sim

            # 3. Compute FK, draw the frames, and compare with the model EEF.
            frames = forward_kinematics(q_dh)
            demo.show(frames)


if __name__ == '__main__':
    main()
