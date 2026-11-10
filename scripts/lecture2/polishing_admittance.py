"""Force-error admittance: planned XY motion + normal contact-force control.
Run: python scripts/lecture2/polishing_admittance.py [--force 10 --force-after 15]
WORLD Z points up; delta is positive down: z_cmd = z_nominal - delta.
Scene, runtime, graphs and export: support/polishing_lab.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from shared.common import pose
from lecture2.support.force_sensor import read_wrench
from lecture2.support.scenes import PolishingLab

DT = .003               # Control period [s].
VIRTUAL_MASS = 10.       # M [kg].
DAMPING = 2000.          # B [Ns/m]; K = 0.
MAX_SPEED = .004         # Maximum normal correction speed [m/s].
MIN_OFFSET, MAX_OFFSET = -.01, .04  # Normal correction limits [m].
CONTACT_HOLD = .3        # Required continuous time within the force tolerance [s].


def options():
    parser = argparse.ArgumentParser(description='Force-error admittance on a flat workpiece.')
    parser.add_argument('--force',type=float,default=10.,help='initial contact-force target [N]')
    parser.add_argument('--force-after',type=float,default=15.,help='force target in the second half [N]')
    parser.add_argument('--duration',type=float,default=14.,help='simulation duration [s]')
    args = parser.parse_args()
    values = [args.force,args.force_after,args.duration]
    if not np.isfinite(values).all() or min(values)<=0:
        parser.error('Force targets and duration must be finite and positive.')
    return args


def tangent_path(rest, elapsed):
    """Time since contact-triggered start; ramp path speed smoothly over one second."""
    elapsed = max(0.,elapsed)
    ramp = min(elapsed,1.)
    phase = (2*np.pi/6.)*(.5*ramp*ramp+max(0.,elapsed-1.))
    phase_rate = (2*np.pi/6.)*ramp
    position = rest+np.array([.015*(1-np.cos(phase)),.06*np.sin(phase),0.])
    velocity = phase_rate*np.array([.015*np.sin(phase),.06*np.cos(phase),0.])
    return position,velocity


def normal_force(lab):
    """F/T sensor -> remove tool weight -> WORLD upward contact-force estimate."""
    force_s, _ = read_wrench(lab.sim, lab.sensor)
    R = pose(lab.sim, lab.sensor)[:3, :3]
    weight_s, _ = lab.payload.wrench(R, lab.gravity)
    force_world = R @ (force_s - weight_s)
    return float(force_world[2])


def normal_step(delta, velocity, force_error):
    """M * delta_ddot + B * delta_dot = F_desired - F_measured; K = 0."""
    acceleration = (force_error - DAMPING * velocity) / VIRTUAL_MASS
    velocity = float(np.clip(velocity + DT * acceleration, -MAX_SPEED, MAX_SPEED))
    candidate = delta + DT * velocity
    delta = float(np.clip(candidate, MIN_OFFSET, MAX_OFFSET))
    if candidate != delta:
        velocity = 0.  # Prevent accumulation beyond the travel limits.
    return delta, velocity


def control(lab, duration, force, force_after):
    delta, velocity, measured = 0., 0., 0.
    stable_time, path_start = 0., None
    for step in range(round(duration / DT)):
        t = lab.elapsed()
        desired = force if t < duration / 2 else force_after

        # 1. READ the contact force; low-pass filter the measurement.
        raw = normal_force(lab)
        measured += DT / (.03 + DT) * (raw - measured)

        # 2. COMPUTE normal motion from force error: too little force -> move down.
        force_error = desired - measured
        delta, velocity = normal_step(delta, velocity, force_error)

        # 3. COMMAND: follow the XY path and add the downward position correction.
        if path_start is None:
            tolerance = max(1., .1 * desired)  # 10% of target, at least 1 N.
            stable_time = stable_time + DT if abs(force_error) <= tolerance else 0.
            if stable_time >= CONTACT_HOLD:
                path_start = lab.path_start = t
                print(f'Contact established at {t:.3f} s; starting XY motion.', flush=True)
        path_time = 0. if path_start is None else t - path_start
        reference, feedforward = tangent_path(lab.rest, path_time)
        reference[2] -= delta
        feedforward[2] = -velocity
        lab.controller.step(reference, feedforward)

        lab.observe(step, t, desired, raw, measured, delta, velocity, reference)
        lab.sim.step()


def main():
    args = options()
    try:
        with PolishingLab(DT) as lab:
            control(lab, args.duration, args.force, args.force_after)
    except KeyboardInterrupt:
        return
    lab.save(args.duration, VIRTUAL_MASS, DAMPING)


if __name__ == '__main__':
    main()
