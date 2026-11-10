"""Curved polishing: local normal force feedback + surface path + tool alignment.
python scripts/lecture2/curved_polishing_admittance.py
Normals come from the known sphere model. No material removal is simulated.
delta > 0 moves INTO the surface: p_cmd = centre + (R + tool_length + gap - delta)*n.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from lecture2.support.robot import Robot, arrow_segments, get_q
from shared.common import connect, pose, stepped_simulation
from lecture2.support.force_sensor import (
    physics, Payload, settle_and_average, ForceSensor, ForceTrackingGraph)
from lecture2.polishing_admittance import DT, VIRTUAL_MASS, DAMPING, normal_step
from lecture2.support.scenes import PolishingScene
from lecture2.support.utils import OUT, save_report


def surface_path(t, amplitude=np.radians(18), period=10.):
    """Meridian sweep; smooth start after contact acquisition at 4 seconds."""
    elapsed = max(0.,t-4.)
    ramp = min(elapsed,1.)
    phase = 2*np.pi/period*(.5*ramp*ramp+max(0.,elapsed-1.))
    theta = amplitude*np.sin(phase)
    theta_dot = amplitude*np.cos(phase)*2*np.pi/period*ramp
    normal = np.array([0.,np.sin(theta),np.cos(theta)])
    normal_dot = theta_dot*np.array([0.,np.cos(theta),-np.sin(theta)])
    # Tool +Z points into the workpiece; retain the initial tool X direction.
    z = -normal
    x = np.array([-1.,0.,0.])
    rotation = np.column_stack([x,np.cross(z,x),z])
    return normal,normal_dot,rotation


def run(sim, k, duration, force, force_after):
    gravity = np.array(sim.getProperty(sim.handle_scene,'gravity'))
    body = pose(sim,k['tool'],k['ft'])
    com = body[:3,:3] @ np.array(sim.getProperty(k['tool'],'centerOfMass'))+body[:3,3]
    payload = Payload(sim.getFloatProperty(k['tool'],'mass'),com)
    sensor = ForceSensor(sim,k['ft'],payload)
    radius,centre = k['surface_radius'],k['surface_center']
    graph = ForceTrackingGraph(sim,k['root'])
    log = []
    with physics(sim,DT), stepped_simulation(sim), graph:
        settle_and_average(sim,k['ft'])
        controller = Robot.from_handles(sim,k).controller(DT)
        delta,velocity,measured = 0.,0.,0.
        trace = sim.addDrawingObject(sim.drawing_lines,4,0,k['root'],10000,[.1,.95,.25])
        arrow = sim.addDrawingObject(sim.drawing_lines,4,0,k['root'],3,[.9,.1,.7])
        previous_contact = None
        start = sim.getSimulationTime()
        for step in range(round(duration/DT)):
            t = sim.getSimulationTime()-start
            normal,normal_dot,tool_rotation = surface_path(t)
            # READ: remove tool weight using ACTUAL sensor orientation; project onto local normal.
            force_s,_ = sensor.read()
            R = pose(sim,k['ft'])[:3,:3]
            weight_s,_ = payload.wrench(R,gravity)
            external_world = R @ (force_s-weight_s)
            raw = float(normal @ external_world)
            measured += DT/(.03+DT)*(raw-measured)
            desired = force if t<duration/2 else force_after
            delta,velocity = normal_step(delta,velocity,desired-measured)
            # COMPUTE: both the surface height and its normal change along the path.
            distance = radius+PolishingScene.pad_bottom+k['clearance']-delta
            reference = centre+distance*normal
            feedforward = distance*normal_dot-velocity*normal
            controller.R_ref = tool_rotation
            controller.step(reference,feedforward)
            graph.update(desired,measured)
            actual = controller.position()
            alignment = np.degrees(np.arccos(np.clip(np.dot(-R[:,2],normal),-1,1)))
            if step%10==0:
                tool_tip = R @ np.array([0,0,PolishingScene.pad_bottom])+actual
                radial = tool_tip-centre; radial /= np.linalg.norm(radial)
                contact = centre+(radius+.001)*radial
                if measured>1. and previous_contact is not None:
                    sim.addDrawingObjectItem(trace,np.r_[previous_contact,contact].tolist())
                previous_contact = contact if measured>1. else None
                sim.addDrawingObjectItem(arrow,None)
                for segment in arrow_segments(contact,.005*max(0.,measured)*normal):
                    sim.addDrawingObjectItem(arrow,segment.tolist())
            log.append(np.r_[t,desired,raw,measured,delta,velocity,reference,actual,normal,
                             np.degrees(np.arcsin(normal[1])),alignment])
            if step%200==0:
                print(f't={t:.1f}s | Fd={desired:.1f} N | Fm={measured:.2f} N | '
                      f'tilt={np.degrees(np.arcsin(normal[1])):+.1f} deg | alignment={alignment:.2f} deg',flush=True)
            sim.step()
        final_q = get_q(sim,k['joints'])
        final_tool = sim.getObjectMatrix(k['tool'],sim.handle_world)
    # Preserve the last tilted pose so the curved-surface alignment stays visible.
    for joint,q in zip(k['joints'],final_q):
        sim.setJointPosition(joint,float(q));sim.setJointTargetPosition(joint,float(q))
    sim.setObjectMatrix(k['tool'],final_tool,sim.handle_world)
    return np.asarray(log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',type=float,default=10.)
    parser.add_argument('--force-after',type=float,default=15.)
    parser.add_argument('--duration',type=float,default=18.)
    args = parser.parse_args()
    if not np.isfinite([args.force,args.force_after,args.duration]).all() or min(args.force,args.force_after,args.duration)<=0:
        parser.error('Force targets and duration must be finite and positive.')
    _,sim = connect()
    k = PolishingScene.setup(sim,curved=True)
    try: log = run(sim,k,args.duration,args.force,args.force_after)
    except KeyboardInterrupt: return
    columns = ['t_s','desired_N','raw_normal_N','measured_N','delta_m','delta_dot_m_s']
    columns += [f'{name}_{a}' for name in ('reference_m','actual_m','normal') for a in 'xyz']
    columns += ['surface_tilt_deg','normal_alignment_error_deg']
    settled = log[log[:,0]>=.75*args.duration]
    save_report('curved_polishing_admittance',dict(samples=log,columns=columns,
        virtual_mass_kg=VIRTUAL_MASS,damping_Ns_m=DAMPING,stiffness_N_m=0.,
        sphere_radius_m=k['surface_radius'],sphere_centre_world=k['surface_center'],
        settled_force_rmse_N=float(np.sqrt(np.mean((settled[:,1]-settled[:,3])**2))),
        note='Known sphere normals; force is projected onto local normal, not WORLD Z. '
             'Actual F/T orientation is used for weight compensation. No inertia compensation/material removal.'))
    np.savetxt(OUT/'curved_polishing_admittance.csv',log,delimiter=',',header=','.join(columns),comments='')
    from lecture2.support.utils import plot_polishing as plot
    plot(log,filename='curved_polishing_admittance.png',curved=True)


if __name__=='__main__': main()
