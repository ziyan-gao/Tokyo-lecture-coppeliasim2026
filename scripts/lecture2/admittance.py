"""Force -> virtual mass/spring/damper -> Cartesian target -> joint POSITION commands.

python scripts/lecture2/admittance.py
python scripts/lecture2/admittance.py --stiffness 0
python scripts/lecture2/admittance.py --interactive   # drag /ForceLab/hand
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from lecture2.support.robot import Robot, matrix, arrow_segments
from shared.common import connect, stepped_simulation, _dummy, trans
from lecture2.support.force_sensor import physics, Payload, settle_and_average, ForceSensor
from lecture2.force_sensor import setup, TOOL_MASS, TOOL_LENGTH
from lecture2.support.utils import OUT, save_report

DT, MASS, DAMPING = .01, 2., 40.  # s, virtual kg, Ns/m
WORKSPACE_RADIUS = .18  # limit the virtual reference, not a guarantee of IK reachability


def admittance_step(x, velocity, force, rest, dt, mass=MASS, damping=DAMPING, stiffness=120.):
    """Semi-implicit Euler: physical force [N] -> virtual velocity [m/s], position [m]."""
    acceleration = (force - damping*velocity - stiffness*(x-rest))/mass
    velocity = velocity + dt*acceleration
    speed = np.linalg.norm(velocity)
    velocity *= min(1., .15/max(speed, 1e-12))
    return x + dt*velocity, velocity


def limit_workspace(x, velocity, rest, radius=WORKSPACE_RADIUS):
    """Project reference onto the demo sphere; remove outward velocity to avoid windup."""
    offset = x-rest
    distance = np.linalg.norm(offset)
    if distance < radius:
        return x, velocity, False
    normal = offset/distance
    x = rest + radius*normal
    velocity = velocity - max(0., np.dot(velocity, normal))*normal
    return x, velocity, True


def run(sim, k, stiffness, duration, interactive=False, mass=MASS, damping=DAMPING):
    gravity = np.array(sim.getProperty(sim.handle_scene, 'gravity'))
    payload = Payload(TOOL_MASS, [0, 0, TOOL_LENGTH/2])
    sensor = ForceSensor(sim,k['ft'],payload)
    with physics(sim, DT), stepped_simulation(sim):
        settle_and_average(sim, k['ft'])
        controller = Robot.from_handles(sim,k).controller(DT)
        rest = controller.position().copy()
        x, velocity, filtered = rest.copy(), np.zeros(3), np.zeros(3)
        hand = _dummy(sim, 'hand', k['root'], trans(*rest), size=.035, color=[.1,.9,.2])
        arrow = sim.addDrawingObject(sim.drawing_lines, 4, 0, -1, 3, [1., .25, .03])
        start = sim.getSimulationTime()
        log = []
        at_boundary = False
        if interactive:
            sim.setObjectSelection([hand])
            print('Drag /ForceLab/hand. Orange arrow = force, not an IK target.\n'
                  'Reference travel is limited to 18 cm; drag back inward to return.', flush=True)
        for _ in range(round(duration/DT)):
            t = sim.getSimulationTime()-start
            # 1. READ: F/T in sensor frame. Compensate weight and rotate to world.
            force_s, _ = sensor.read()
            R = matrix(sim, k['ft'])[:3, :3]
            weight_s, _ = payload.wrench(R, gravity)
            external = R @ (force_s-weight_s)
            filtered += (DT/(.04+DT))*(external-filtered)
            # 2. COMPUTE: admittance produces motion; inverse dynamics is unnecessary.
            # Isotropic matrices: M = mass*I, B = damping*I, K = stiffness*I.
            x, velocity = admittance_step(x, velocity, filtered, rest, DT,
                                         mass=mass, damping=damping, stiffness=stiffness)
            x, velocity, limited = limit_workspace(x, velocity, rest)
            if limited and not at_boundary:
                print('18 cm reference boundary reached; motion limited. Drag inward to return.', flush=True)
            at_boundary = limited
            # 3. WRITE: differential IK -> position targets -> one physics step.
            actual = controller.step(x, velocity)
            if interactive:
                hand_pos = np.array(sim.getObjectPosition(hand, sim.handle_world))
                applied = 100.*(hand_pos-actual)
                applied *= min(1., 15./max(np.linalg.norm(applied), 1e-12))
            else:
                applied = np.array([0., 8. if .5 <= t < 2.5 else 0., 0.])
            sim.addForceAndTorque(k['tool'], applied.tolist(), [0.,0.,0.])
            sim.addDrawingObjectItem(arrow, None)
            for segment in arrow_segments(actual, .01*applied):
                sim.addDrawingObjectItem(arrow, segment.tolist())
            log.append(np.r_[t, applied, external, filtered, x-rest, actual-rest])
            sim.step()
    return np.array(log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stiffness', type=float, default=120., help='N/m; 0 enables hand guiding')
    parser.add_argument('--mass', type=float, default=MASS, help='virtual mass M=mI [kg]')
    parser.add_argument('--damping', type=float, default=DAMPING, help='virtual damping B=bI [Ns/m]')
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    if not np.all(np.isfinite([args.mass, args.damping, args.stiffness, args.duration])) or args.mass <= 0 or args.damping < 0 or args.stiffness < 0 or args.duration <= 0:
        parser.error('mass/duration must be positive; damping/stiffness nonnegative; all finite')
    _, sim = connect()
    k = setup(sim)
    try:
        log = run(sim, k, args.stiffness, 600 if args.interactive else args.duration,
                  args.interactive, args.mass, args.damping)
    except KeyboardInterrupt:
        return
    columns = ['t_s']+[f'{name}_{axis}' for name in
        ('applied_N', 'external_N', 'filtered_N', 'reference_m', 'actual_m') for axis in 'xyz']
    save_report('admittance', dict(mass_kg=args.mass, damping_Ns_m=args.damping,
        stiffness_N_m=args.stiffness, columns=columns, samples=log,
        note='World-frame translation; fixed orientation; position-controlled inner loop.'))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT/'admittance.csv', log, delimiter=',', header=','.join(columns), comments='')
    from lecture2.support.utils import plot_admittance as plot
    plot(log, args.stiffness)
    print('Peak displacement [mm]:', np.max(np.abs(log[:,13:16]),axis=0)*1000)


if __name__ == '__main__':
    main()
