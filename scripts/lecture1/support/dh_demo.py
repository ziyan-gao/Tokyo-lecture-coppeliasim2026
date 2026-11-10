"""CoppeliaSim setup, drawings and comparison for the DH lecture."""
import math
import os
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class DHScene:
    def __init__(self, marker_interval=.05):
        self.marker_interval = marker_interval
        self.started = False
        self.chain, self.axes = None, []
        self.last_errors = None

    def __enter__(self):
        client = RemoteAPIClient(host=os.getenv('COPPELIASIM_HOST','localhost'),
                                 port=int(os.getenv('COPPELIASIM_PORT','23000')))
        self.sim = sim = client.require('sim')
        if sim.getSimulationState()!=sim.simulation_stopped:
            raise RuntimeError('Stop simulation before running this demo')
        try:
            self.setup_robot()
            self.setup_drawings()
            self.next_sample, self.next_report = 0., 0.
            sim.setStepping(True)
            sim.startSimulation()
            self.started = True
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def setup_robot(self):
        sim = self.sim
        self.robot = sim.getObject('/LBRiiwa14R820',{'noError':True})
        if self.robot==-1:
            self.robot = sim.loadModel(sim.getStringParam(sim.stringparam_resourcesdir)+
                '/models/robots/non-mobile/KUKA LBR iiwa 14 R820.ttm')
        path = sim.getObjectAlias(self.robot,1)
        self.joints = [sim.getObject(path+'/joint',{'index':i}) for i in range(7)]
        self.eef = sim.getObject(path+'/connection')
        script = sim.getScript(sim.scripttype_simulation,self.robot)
        if script!=-1: sim.setScriptInt32Param(script,sim.scriptintparam_enabled,0)
        for joint in self.joints:
            sim.setJointMode(joint,sim.jointmode_dynamic,0)
            sim.setIntProperty(joint,'dynCtrlMode',sim.jointdynctrl_position)
        old = sim.getObject(path+'/DH_FK_Demo',{'noError':True})
        if old!=-1: sim.removeObjects(sim.getObjectsInTree(old))
        self.root = sim.createDummy(.01)
        sim.setObjectAlias(self.root,'DH_FK_Demo')
        # DH origin is at the base mounting plane. Remove the current joint-1
        # rotation so this fixed base frame is independent of the starting pose.
        base = np.array(sim.getObjectMatrix(self.joints[0],self.robot)).reshape(3,4)
        q = sim.getJointPosition(self.joints[0])
        c,s = math.cos(q),math.sin(q)
        base[:,:3] = base[:,:3] @ np.array([[c,s,0],[-s,c,0],[0,0,1]])
        base[2,3] = sim.getObjectFloatParam(self.robot,sim.objfloatparam_objbbox_min_z)
        sim.setObjectParent(self.root,self.robot,True)
        sim.setObjectMatrix(self.root,base.ravel().tolist(),self.robot)

    def setup_drawings(self):
        sim = self.sim
        flags = sim.drawing_local | sim.drawing_painttag | sim.drawing_overlay
        self.chain = sim.addDrawingObject(sim.drawing_lines|flags,5,0,self.root,7,[1,.65,.15])
        self.axes = [sim.addDrawingObject(sim.drawing_lines|flags,3,0,self.root,8,color)
                     for color in ([1,0,0],[0,1,0],[0,0,1])]
        self.trail = sim.addDrawingObject(sim.drawing_triangles|flags,1,0,self.root,10000,[.1,.8,1])
        self.triangle = np.array([[.01,0,0],[-.005,.005,0],[-.005,-.005,0]])

    def show(self, frames):
        sim,T = self.sim,frames[-1]
        sim.addDrawingObjectItem(self.chain,None)
        for a,b in zip(frames[:-1],frames[1:]):
            sim.addDrawingObjectItem(self.chain,np.r_[a[:3,3],b[:3,3]].tolist())
        for i,drawing in enumerate(self.axes):
            sim.addDrawingObjectItem(drawing,None)
            for frame in frames:
                length = .16 if frame is T else .08
                sim.addDrawingObjectItem(drawing,np.r_[frame[:3,3],frame[:3,3]+length*frame[:3,i]].tolist())
        t = sim.getSimulationTime()
        if t>=self.next_sample:
            points = self.triangle @ T[:3,:3].T+T[:3,3]
            sim.addDrawingObjectItem(self.trail,points.ravel().tolist())
            self.next_sample = t+self.marker_interval
        if t>=self.next_report:
            actual = np.array(sim.getObjectMatrix(self.eef,self.root)).reshape(3,4)
            error = np.linalg.norm(T[:3,3]-actual[:,3])*1000
            angle = math.degrees(math.acos(np.clip((np.trace(T[:3,:3].T@actual[:,:3])-1)/2,-1,1)))
            self.last_errors = error,angle
            print(f't={t:.2f}s | EEF error={error:.3f} mm, {angle:.3f} deg\n{T.round(4)}',flush=True)
            self.next_report = t+1.
        time.sleep(.02)

    def __exit__(self, *exc):
        if self.started:
            self.sim.stopSimulation()
            self.sim.setStepping(False)
            while self.sim.getSimulationState()!=self.sim.simulation_stopped:
                time.sleep(.05)
        for drawing in ([self.chain] if self.chain is not None else [])+self.axes:
            self.sim.addDrawingObjectItem(drawing,None)
            # Triangle pose history stays visible after the robot resets.
