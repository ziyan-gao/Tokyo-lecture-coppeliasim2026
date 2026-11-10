"""Scene, panel, input handling and drawings for manual numerical IK lessons."""
import math
import os
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from lecture1.jacobian_demo import HOME, arrow_segments, geometric_jacobian

NORMAL_GOAL = np.radians([30, 10, 20, -40, -20, 50, 35])


class IKPanel:
    """UI only: the supplied error function and external solver own the IK math."""
    def __init__(self, error_function, position_tol=.001, angle_tol=math.radians(.5), pinv_rcond=1e-6, solver="manual"):
        self.solver = solver
        self.error_function = error_function
        self.pinv_rcond = pinv_rcond
        self._last_diagnostic_poses = None
        self.position_tol, self.angle_tol = position_tol, angle_tol
        self.iteration = 0
        self.client = RemoteAPIClient(host=os.getenv('COPPELIASIM_HOST', 'localhost'),
                                      port=int(os.getenv('COPPELIASIM_PORT', '23000')))
        self.sim, self.api = self.client.require('sim'), self.client.require('simUI')
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('Stop simulation before running this numerical IK demo')
        robot = sim.getObject('/LBRiiwa14R820', {'noError': True})
        if robot == -1:
            robot = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir) +
                                  '/models/robots/non-mobile/KUKA LBR iiwa 14 R820.ttm')
        self.robot = robot
        path = sim.getObjectAlias(robot, 1)
        self.joints = [sim.getObject(path+'/joint', {'index': i}) for i in range(7)]
        self.eef = sim.getObject(path+'/connection')
        demo = sim.getScript(sim.scripttype_simulation, robot)
        if demo != -1:
            sim.setScriptInt32Param(demo, sim.scriptintparam_enabled, 0)
        self.lower, self.upper = np.full(7,-np.inf), np.full(7,np.inf)
        for i,(joint,q) in enumerate(zip(self.joints,HOME)):
            sim.setJointPosition(joint,float(q))
            sim.setJointTargetPosition(joint,float(q))
            cyclic,interval = sim.getJointInterval(joint)
            if not cyclic:
                self.lower[i], self.upper[i] = interval[0], interval[0]+interval[1]
        old = sim.getObject('/JacobianIK', {'noError': True})
        if old != -1:
            sim.removeObjects(sim.getObjectsInTree(old))
        self.root = sim.createDummy(.001)
        sim.setObjectAlias(self.root,'JacobianIK')
        try:
            self.target = sim.createDummy(.035)
            sim.setObjectAlias(self.target,'IK_Target')
            sim.setObjectParent(self.target,self.root,True)
            self.set_target(self.pose(self.eef))
            self._create_panel()
            flags = sim.drawing_lines | sim.drawing_local | sim.drawing_painttag | sim.drawing_overlay
            self.axes = [[sim.addDrawingObject(flags,width,0,self.root,1,color) for color in colors]
                         for width,colors in [(4,[[1,0,0],[0,1,0],[0,0,1]]),
                                               (2,[[1,.6,.6],[.6,1,.6],[.6,.6,1]])]]
            self.error_arrow = sim.addDrawingObject(flags,3,0,self.root,3,[1,.4,0])
            self.trail = sim.addDrawingObject(flags,2,0,self.root,10000,[0,.8,1])
            self._fill_fields()
            self.api.setLabelText(self.ui,31,'Scene stays stopped. Apply target, then click once per IK iteration.')
            self.refresh()
        except Exception:
            self.close()
            raise

    def _create_panel(self):
        fields = ''.join(f'<label text="{name}"/><edit id="{i}" value="0"/>'
                         for i,name in enumerate(('X (m)','Y (m)','Z (m)',
                                                   'Alpha (deg)','Beta (deg)','Gamma (deg)'),1))
        xml = f'''<ui title="Jacobian IK | One click, one iteration" closeable="false" layout="vbox">
<label text="WORLD target pose | Euler R = Rx(alpha) Ry(beta) Rz(gamma)"/>
<group layout="form">{fields}</group>
<button text="Apply target" on-click="applyTarget"/>
<button text="One IK iteration" on-click="iterateOnce"/>
<button text="Copy current EEF pose to target" on-click="copyCurrent"/>
<group layout="hbox">
<button text="Normal example" on-click="normalExample"/>
<button text="Straight-arm singularity" on-click="singularExample"/>
</group>
<group layout="hbox">
<button text="Near-singular example" on-click="nearExample"/>
<button text="Small bend (keep target)" on-click="bendExample"/>
</group>
<label id="33" text=""/><label id="34" text=""/><label id="35" text=""/>
<label text="error = [p_target - p_current; rotvec(R_target R_current^T)]"/>
<label text="dq = gain * pinv(J) * error; then limit the joint step"/>
<label text="RGB: target | pale RGB: current | cyan: EEF path | orange: position error"/>
<label id="30" text=""/><label id="31" text=""/><label id="32" text=""/>
<button text="Exit" on-click="exitDemo"/></ui>'''
        if self.solver == 'simIK':
            xml = xml.replace('Jacobian IK | One click, one iteration', 'simIK | Configure and solve IK')
            xml = xml.replace('One IK iteration', 'Solve with simIK')
            xml = xml.replace('dq = gain * pinv(J) * error; then limit the joint step',
                              'Scene -> syncFromSim -> handleGroup -> apply solved joint angles')
            xml = xml.replace('error = [p_target - p_current; rotvec(R_target R_current^T)]',
                              'One solve call can perform multiple internal IK iterations.')
        code = """local api = require('simUI')
local action = ''
function sysCall_init() ui = api.create([=[%s]=]) end
function getPanel() return ui end
function applyTarget() action = 'apply' end
function iterateOnce() action = 'iterate' end
function copyCurrent() action = 'copy' end
function exitDemo() action = 'exit' end
function normalExample() action = 'normal' end
function singularExample() action = 'singular' end
function nearExample() action = 'near' end
function bendExample() action = 'bend' end
function takeAction() local a = action; action = ''; return a end
function sysCall_cleanup() if ui then api.destroy(ui) end end
""" % xml
        self.script = self.sim.createScript(self.sim.scripttype_customization,code,0,'lua')
        self.sim.setObjectParent(self.script,self.root,True)
        self.sim.initScript(self.script)
        self.ui = self.sim.callScriptFunction('getPanel',self.script)

    def pose(self, handle):
        T = np.eye(4)
        T[:3] = np.array(self.sim.getObjectMatrix(handle,self.sim.handle_world)).reshape(3,4)
        return T

    def set_target(self, T):
        self.sim.setObjectMatrix(self.target,T[:3].ravel().tolist(),self.sim.handle_world)

    def _fill_fields(self):
        values = self.sim.getObjectPosition(self.target,self.sim.handle_world) + list(
            np.degrees(self.sim.getObjectOrientation(self.target,self.sim.handle_world)))
        for i,value in enumerate(values,1):
            self.api.setEditValue(self.ui,i,f'{value:.5f}',True)

    def poll(self):
        """Handle target editing; return 'iterate', 'exit', or an empty action."""
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            raise RuntimeError('Keep simulation stopped for numerical IK')
        action = self.sim.callScriptFunction('takeAction',self.script)
        if action in ('normal', 'singular', 'near', 'bend'):
            self.set_example(action)
        if action == 'copy':
            self.set_target(self.pose(self.eef))
            self._fill_fields()
        if action == 'apply':
            try:
                values = np.array([float(self.api.getEditValue(self.ui,i)) for i in range(1,7)])
                if not np.isfinite(values).all():
                    raise ValueError()
                T = self.sim.buildMatrix(values[:3].tolist(),np.radians(values[3:]).tolist())
                self.sim.setObjectMatrix(self.target,T,self.sim.handle_world)
                self.api.setLabelText(self.ui,31,'Target applied. Click One IK iteration.')
                print('Target homogeneous matrix (world):\n',self.pose(self.target),flush=True)
            except ValueError:
                self.api.setLabelText(self.ui,31,'Invalid target: enter six finite numbers.')
        self.refresh()
        if not action:
            time.sleep(.08)
        return action

    def set_example(self, example):
        def configure(q):
            for joint, value in zip(self.joints, q):
                self.sim.setJointPosition(joint, float(value))
                self.sim.setJointTargetPosition(joint, float(value))

        def bent_pose(degrees):
            # iiwa14: two arm lengths .42 and .40 m. Cancel lateral motion
            # and restore wrist orientation, yielding a reachable lower target.
            a = math.radians(degrees)
            b = math.asin(.42/.40*math.sin(a))
            return [0, a, 0, a+b, 0, b, 0]

        if example == 'normal':
            # Generate a known reachable pose with substantial position/rotation differences.
            configure(NORMAL_GOAL)
            target = self.pose(self.eef)
            configure(HOME)
            self.set_target(target)
            message = 'Distant reachable target with different orientation. Click iterations to approach it.'
        elif example == 'bend':
            configure(bent_pose(5))
            message = 'Initial guess changed; target unchanged. Try the same pseudoinverse again.'
        else:
            configure(bent_pose(20))
            self.set_target(self.pose(self.eef))  # Known reachable configuration!
            configure(np.zeros(7) if example == 'singular' else bent_pose(1))
            message = ('Reachable target below the straight arm. Try an iteration: first-order motion is missing.'
                       if example == 'singular' else
                       'Near singularity: inspect small singular values and the large raw pseudoinverse step.')
        if example != 'bend':
            self.iteration = 0
            self.sim.addDrawingObjectItem(self.trail, None)
        self._fill_fields()
        self.api.setLabelText(self.ui,31,message)
        self.api.setLabelText(self.ui,32,'')
        self._last_diagnostic_poses = None

    def _diagnostics(self, current, desired):
        if self.solver == 'simIK':
            return
        if self._last_diagnostic_poses is not None:
            old_current, old_desired = self._last_diagnostic_poses
            if np.allclose(current,old_current,atol=1e-10,rtol=0) and np.allclose(desired,old_desired,atol=1e-10,rtol=0):
                return
        J = geometric_jacobian(self.sim,self.joints,self.eef)
        s = np.linalg.svd(J,compute_uv=False)
        cutoff = self.pinv_rcond*s[0]
        rank = int(np.sum(s>cutoff))
        error = self.error_function(current,desired)
        raw_dq = np.linalg.pinv(J,rcond=self.pinv_rcond) @ error
        residual = error-J@raw_dq
        status = 'SINGULAR (numerical rank deficient)' if rank<6 else 'FULL ROW RANK'
        self.api.setLabelText(self.ui,33,f'Current J: rank {rank}/6 | {status} | pinv rcond={self.pinv_rcond:g}')
        self.api.setLabelText(self.ui,34,'Singular values (unscaled J): '+np.array2string(s,precision=2))
        self.api.setLabelText(self.ui,35,
            f'Uncorrectable by current linearization: {np.linalg.norm(residual[:3])*1000:.2f} mm, '
            f'{math.degrees(np.linalg.norm(residual[3:])):.3f} deg | '
            f'Raw pinv step (before gain/cap): max {np.max(np.abs(np.degrees(raw_dq))):.2f} deg')
        self._last_diagnostic_poses = (current.copy(),desired.copy())

    def refresh(self):
        current,desired = self.pose(self.eef),self.pose(self.target)
        self._diagnostics(current,desired)
        for handles,T in zip(self.axes,[desired,current]):
            for i,h in enumerate(handles):
                self.sim.addDrawingObjectItem(h,None)
                self.sim.addDrawingObjectItem(h,np.r_[T[:3,3],T[:3,3]+.12*T[:3,i]].tolist())
        self.sim.addDrawingObjectItem(self.error_arrow,None)
        for line in arrow_segments(current[:3,3],desired[:3,3]-current[:3,3]):
            self.sim.addDrawingObjectItem(self.error_arrow,line.tolist())
        error = self.error_function(current,desired)
        distance,angle = np.linalg.norm(error[:3]),np.linalg.norm(error[3:])
        state = 'CONVERGED' if distance<self.position_tol and angle<self.angle_tol else 'Ready'
        counter = 'Solve call' if self.solver == 'simIK' else 'Iteration'
        self.api.setLabelText(self.ui,30,f'{counter} {self.iteration} | position {distance*1000:.3f} mm | '
                                       f'rotation {math.degrees(angle):.3f} deg | {state}')

    def report_iteration(self, previous, before, dq, after, solve_ms, elapsed_ms):
        self.iteration += 1
        current = self.pose(self.eef)
        self.sim.addDrawingObjectItem(self.trail,np.r_[previous[:3,3],current[:3,3]].tolist())
        self.api.setLabelText(self.ui,31,f'Before: e_p = {before[:3].round(4)} m; e_R = {before[3:].round(4)} rad')
        self.api.setLabelText(self.ui,32,f'Applied dq (deg): {np.degrees(dq).round(3)} | '
                                       f'solve {solve_ms:.3f} ms | update {elapsed_ms:.1f} ms')
        self.refresh()
        print(f'Iteration {self.iteration}: position {np.linalg.norm(before[:3])*1000:.3f} -> '
              f'{np.linalg.norm(after[:3])*1000:.3f} mm; orientation '
              f'{math.degrees(np.linalg.norm(before[3:])):.3f} -> '
              f'{math.degrees(np.linalg.norm(after[3:])):.3f} deg | solve {solve_ms:.3f} ms',flush=True)

    def close(self):
        if self.sim.isHandle(self.root):
            self.sim.removeObjects(self.sim.getObjectsInTree(self.root))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
