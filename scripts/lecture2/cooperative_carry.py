"""Human-guided carrying: handle force -> F/T sensor -> admittance -> position servos.
python scripts/lecture2/cooperative_carry.py           # drag the red wireframe handle
python scripts/lecture2/cooperative_carry.py --demo    # scripted operator force, same controller
The green I-profile is rigidly held. The grey socket marks the assembly goal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
from lecture2.support.robot import Robot, arrow_segments, get_q
from shared.common import connect, pose, stepped_simulation
from lecture2.support.force_sensor import physics, Payload, settle_and_average, ForceSensor
from lecture2.admittance import admittance_step, limit_workspace
from lecture2.force_sensor import limit
from lecture2.support.scenes import CarryScene
from lecture2.support.utils import save_report

DT, MASS, DAMPING = .01, 2., 45.
HANDLE_STIFFNESS, MAX_FORCE = 150., 12.


def run(sim, k, duration, demo=False):
    gravity = np.array(sim.getProperty(sim.handle_scene,'gravity'))
    body = pose(sim,k['tool'],k['ft'])
    com = body[:3,:3] @ np.array(sim.getProperty(k['tool'],'centerOfMass'))+body[:3,3]
    payload = Payload(sim.getFloatProperty(k['tool'],'mass'),com)
    sensor = ForceSensor(sim,k['ft'],payload)
    log, reached = [], False
    navigation = sim.getNavigationMode()
    try:
        with physics(sim,DT), stepped_simulation(sim):
            settle_and_average(sim,k['ft'])
            controller = Robot.from_handles(sim,k).controller(DT)
            rest = controller.position().copy()
            goal = CarryScene.goal_fixture(sim,k,rest)
            x,velocity,filtered = rest.copy(),np.zeros(3),np.zeros(3)
            arrow = CarryScene.drawing(sim,-1)
            sim.setObjectSelection([k['hand']])
            sim.setNavigationMode(sim.navigation_objectshift | sim.navigation_clickselection |
                sim.navigation_camerazoomwheel | sim.navigation_camerarotatemiddlebutton)
            print('Drag /ForceLab/Drag_Handle towards the grey socket; release to stop. Ctrl+C exits.',flush=True)
            start,deadline = sim.getSimulationTime(),time.perf_counter()
            for step in range(round(duration/DT)):
                if sim.getSimulationState()==sim.simulation_stopped: break
                t = sim.getSimulationTime()-start
                # READ: sensor force includes tool weight; remove it and rotate into WORLD.
                force_s,_ = sensor.read()
                R = pose(sim,k['ft'])[:3,:3]
                weight_s,_ = payload.wrench(R,gravity)
                external = R @ (force_s-weight_s)
                filtered += DT/(.04+DT)*(external-filtered)
                # COMPUTE: K=0 permits hand guiding without returning to the start position.
                x,velocity = admittance_step(x,velocity,filtered,rest,DT,
                                            mass=MASS,damping=DAMPING,stiffness=0.)
                x,velocity,_ = limit_workspace(x,velocity,rest)
                actual = controller.step(x,velocity)
                # Mouse displacement is a virtual hand spring, NOT a robot position command.
                anchor = np.array(sim.getObjectPosition(k['handle'],sim.handle_world))
                if demo:
                    applied = limit(120.*(goal-actual)-12.*velocity,MAX_FORCE) if .5<t<6. else np.zeros(3)
                    sim.setObjectPose(k['hand'],sim.getObjectPose(k['handle'],sim.handle_world),sim.handle_world)
                elif sim.getInt32Param(sim.intparam_mouse_buttons) & 1:
                    hand = np.array(sim.getObjectPosition(k['hand'],sim.handle_world))
                    applied = limit(HANDLE_STIFFNESS*(hand-anchor),MAX_FORCE)
                else:
                    applied = np.zeros(3)
                    sim.setObjectPose(k['hand'],sim.getObjectPose(k['handle'],sim.handle_world),sim.handle_world)
                CarryScene.force_at_handle(sim,k,applied)
                if step%5==0:
                    sim.addDrawingObjectItem(arrow,None)
                    for segment in arrow_segments(anchor,.015*applied):
                        sim.addDrawingObjectItem(arrow,segment.tolist())
                error = np.linalg.norm(actual-goal)
                if error<.012 and not reached:
                    reached = True
                    sim.setShapeColor(k['goal_marker'],None,sim.colorcomponent_ambient_diffuse,[.1,.9,.2])
                    print(f'Goal reached at {t:.2f} s (position error {error*1000:.1f} mm).',flush=True)
                log.append(np.r_[t,applied,filtered,actual,error])
                sim.step()
                if not demo:
                    deadline = max(deadline+DT,time.perf_counter()-DT)
                    time.sleep(max(0.,deadline-time.perf_counter()))
            final_error = np.linalg.norm(controller.position()-goal)
            return dict(samples=np.asarray(log),goal_reached=reached,final_error_m=final_error,
                goal_world=goal,virtual_mass_kg=MASS,damping_Ns_m=DAMPING,stiffness_N_m=0.,
                final_joint_rad=get_q(sim,k['joints']),final_tool_matrix=sim.getObjectMatrix(k['tool'],sim.handle_world),
                tool_mass_kg=payload.mass,mode='scripted operator force' if demo else 'mouse-guided handle',
                columns=['t_s']+[f'{n}_{a}' for n in ('applied_N','external_N','actual_m') for a in 'xyz']+['goal_error_m'])
    finally:
        sim.setNavigationMode(navigation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true',help='use a scripted operator force for a repeatable demonstration')
    parser.add_argument('--duration',type=float,help='simulation seconds; default 8 in demo, 600 interactively')
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (8. if args.demo else 600.)
    if not np.isfinite(duration) or duration<=0: parser.error('Duration must be positive and finite.')
    _,sim = connect()
    k = CarryScene.setup(sim)
    try:
        result = run(sim,k,duration,args.demo)
    except KeyboardInterrupt:
        return
    # Keep the completed pose visible after stopping physics.
    for joint,value in zip(k['joints'],result['final_joint_rad']):
        sim.setJointPosition(joint,float(value))
        sim.setJointTargetPosition(joint,float(value))
    sim.setObjectMatrix(k['tool'],result['final_tool_matrix'],sim.handle_world)
    sim.setObjectPose(k['hand'],sim.getObjectPose(k['handle'],sim.handle_world),sim.handle_world)
    if not sim.isHandle(k['goal_marker']):
        CarryScene.goal_fixture(sim,k,np.array(result['goal_world'])-CarryScene.goal_offset)
    if result['goal_reached']:
        sim.setShapeColor(k['goal_marker'],None,sim.colorcomponent_ambient_diffuse,[.1,.9,.2])
    save_report('cooperative_carry',result)
    print(f"Final goal error: {result['final_error_m']*1000:.1f} mm")


if __name__=='__main__':
    main()
