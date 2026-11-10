"""F/T readings and tool-weight compensation; native graphs share one viewer factory."""

from contextlib import contextmanager

import numpy as np

from .robot import matrix


class ForceSensor:
    """Raw sensor-frame wrench, with optional model-based tool-weight compensation."""

    def __init__(self, sim, handle, payload=None):
        self.sim, self.handle = sim, handle
        self.payload = payload if payload is not None else Payload()

    def read(self):
        return read_wrench(self.sim, self.handle)

    def external(self, gravity):
        return self.payload.external(self.sim, self.handle, gravity)

    def settle(self, settle=0.5, average=0.1):
        return settle_and_average(self.sim, self.handle, settle, average)


class Payload:
    """Tool mass m and centre of mass c (sensor frame). A static reading at sensor rotation R is
        F = m R^T g,     T = c x F
    so with several poses, m and c follow from two linear least-squares problems."""

    def __init__(self, mass=0.0, com=(0, 0, 0)):
        self.mass, self.com = mass, np.asarray(com, float)

    @classmethod
    def identify(cls, readings, rotations, g):
        a = np.array([R.T @ g for R in rotations])  # gravity seen in the sensor frame
        F, T = readings[:, :3], readings[:, 3:]
        mass = float(np.sum(F * a) / np.sum(a * a))  # min |F - m a|^2
        skew = lambda f: np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]])
        A = np.vstack([-skew(f) for f in F])  # c x F = -[F]x c
        com = np.linalg.lstsq(A, T.ravel(), rcond=None)[0]
        return cls(mass, com)

    def wrench(self, R, g):
        """What the tool's own weight alone makes the sensor read (sensor frame)."""
        F = self.mass * R.T @ g
        return F, np.cross(self.com, F)

    def external(self, sim, ft, g):
        """Measured minus own weight = the force the WORLD exerts on the tool, in the world frame."""
        F, T = read_wrench(sim, ft)
        R = matrix(sim, ft)[:3, :3]
        F0, T0 = self.wrench(R, g)
        return R @ (F - F0), R @ (T - T0)


def read_wrench(sim, sensor):
    """Force [N], torque [Nm] about the sensor origin, expressed in SENSOR axes."""
    state, force, torque = sim.readForceSensor(sensor)
    if not state & 1:
        raise RuntimeError("F/T sample unavailable: step physics and fill the sensor filter first.")
    return np.asarray(force, float), np.asarray(torque, float)


def settle_and_average(sim, ft, settle=0.5, average=0.1):
    """Average a stationary wrench, using the actual simulation time step."""
    dt = sim.getSimulationTimeStep()
    for _ in range(round(settle / dt)):
        sim.step()
    samples = []
    for _ in range(max(1, round(average / dt))):
        sim.step()
        samples.append(np.r_[read_wrench(sim, ft)])
    return np.mean(samples, axis=0)


@contextmanager
def physics(sim, dt=0.005, engine=None):
    """MuJoCo, and one physics step per sim.step(): the torque we write is held for exactly dt,
    which is what our integrator assumes too. Restores the scene settings afterwards
    (the other lessons run at the default 50 ms)."""
    engine = sim.physics_mujoco if engine is None else engine
    old = (
        sim.getInt32Param(sim.intparam_dynamic_engine),
        sim.getFloatProperty(sim.handle_scene, "timeStep"),
        sim.getFloatProperty(sim.handle_scene, "dynamicsStepSize"),
    )
    sim.setInt32Param(sim.intparam_dynamic_engine, engine)
    sim.setFloatProperty(sim.handle_scene, "timeStep", dt)
    sim.setFloatProperty(sim.handle_scene, "dynamicsStepSize", dt)
    try:
        yield
    finally:
        sim.setInt32Param(sim.intparam_dynamic_engine, old[0])
        sim.setFloatProperty(sim.handle_scene, "timeStep", old[1])
        sim.setFloatProperty(sim.handle_scene, "dynamicsStepSize", old[2])


class _NativeGraphs:
    """Shared native graph construction, Lua viewer attachment and final curve snapshots."""

    def __init__(self, sim, root):
        self.sim, self.root = sim, root
        self.items, self.labels = [], []

    def _graph(self, name, symbol, unit, colors=None, labels=None):
        sim = self.sim
        xml = """<CoppeliaSim><filetype>simpleModel</filetype><xmlSerializationNb>1</xmlSerializationNb>
<graph><size>0.02</size><bufferSize>20000</bufferSize><switches>
<cyclic>true</cyclic><explicitHandling>true</explicitHandling></switches></graph></CoppeliaSim>"""
        code = """local sim=require('sim')
function build()
    local temporary=os.tmpname();os.remove(temporary)
    local path=temporary..'.simmodel.xml'
    local file=assert(io.open(path,'w'));file:write([=[@INSERT@]=]);file:close()
    local ok,result=pcall(sim.loadModel,path);os.remove(path);assert(ok,result)
end""".replace("@INSERT@", xml)
        helper = sim.createScript(sim.scripttype_customization, code, 0, "lua")
        sim.setObjectParent(helper, self.root, True)
        sim.initScript(helper)
        try:
            before = set(sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_graph, 0))
            sim.callScriptFunction("build", helper)
            created = set(sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_graph, 0)) - before
            if len(created) != 1:
                raise RuntimeError("Expected one native Graph from XML import.")
            graph = created.pop()
            sim.setObjectParent(graph, self.root, True)
        finally:
            sim.removeObjects([helper])
        sim.setObjectAlias(graph, name)
        sim.setExplicitHandling(graph, 1)
        sim.setIntProperty(graph, "layer", 0)
        colors = colors if colors is not None else ([0.9, 0.08, 0.08], [0.05, 0.65, 0.15], [0.08, 0.3, 1.0])
        labels = labels if labels is not None else [symbol + axis.lower() for axis in "XYZ"]
        streams = [sim.addGraphStream(graph, label, unit, 0, color) for label, color in zip(labels, colors)]
        return graph, streams

    def _viewer(self, graph, callbacks):
        code = """local sim=require('sim')
local graph=require('models.graph_customization-2')
function sysCall_init()
    sim.writeCustomBufferData(sim.getObject('..'),'ABC_GRAPH_INFO',sim.packTable({bitCoded=7,updateFreq=0}))
    graph.init()
end
@INSERT@
function sysCall_cleanup() if graph.cleanup then graph.cleanup() end end
""".replace("@INSERT@", callbacks)
        viewer = self.sim.createScript(self.sim.scripttype_customization, code, 0, "lua")
        self.sim.setObjectParent(viewer, graph, True)
        self.sim.setObjectAlias(viewer, "native_graph_viewer")
        self.sim.initScript(viewer)
        return viewer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for (graph, streams), labels in zip(self.items, self.labels):
            for stream, label in zip(streams, labels):
                self.sim.duplicateGraphCurveToStatic(graph, stream, label)


class FilteredWrenchGraphs(_NativeGraphs):
    """Raw/filtered XYZ curves. Python reads and filters; Lua only refreshes the viewer."""

    def __init__(self, sim, root):
        super().__init__(sim, root)
        self.viewers, self.next_refresh = [], 0.0
        pale = ([0.95, 0.60, 0.60], [0.50, 0.80, 0.60], [0.55, 0.65, 1.0])
        saturated = ([0.9, 0.08, 0.08], [0.05, 0.65, 0.15], [0.08, 0.3, 1.0])
        for name, symbol, unit in (("Force_RAW_LPF_SENSOR", "F", "N"), ("Torque_RAW_LPF_SENSOR", "T", "Nm")):
            graph, streams = self._graph(name, symbol + "_raw_", unit, colors=pale)
            streams += [
                sim.addGraphStream(graph, symbol + "_LPF_" + axis, unit, 0, color)
                for axis, color in zip("xyz", saturated)
            ]
            self.items.append((graph, streams))
            self.labels.append(
                ["Recorded " + symbol + " " + kind + " " + axis for kind in ("raw", "LPF") for axis in "xyz"]
            )
            self.viewers.append(self._viewer(graph, "function refresh() graph.updateCurves() end"))

    def update(self, raw, filtered):
        """Called AFTER sim.step and sensor reading: plot matching raw / LPF samples."""
        timestamp = self.sim.getSimulationTime()
        for i, (graph, streams) in enumerate(self.items):
            offset = 3 * i
            values = list(raw[offset : offset + 3]) + list(filtered[offset : offset + 3])
            for stream, value in zip(streams, values):
                self.sim.setGraphStreamValue(graph, stream, float(value))
            self.sim.handleGraph(graph, timestamp)
        if timestamp >= self.next_refresh:
            for viewer in self.viewers:
                self.sim.callScriptFunction("refresh", viewer)
            self.next_refresh = timestamp + 0.05


class ForceTrackingGraph(_NativeGraphs):
    """Desired and measured normal-force curves, fed from the Python control loop."""

    def __init__(self, sim, root):
        super().__init__(sim, root)
        self.graph, self.streams = self._graph(
            "Normal_contact_force",
            "F",
            "N",
            colors=([0.95, 0.35, 0.06], [0.05, 0.7, 0.2]),
            labels=["Desired contact force", "Measured contact force"],
        )
        self.items.append((self.graph, self.streams))
        self.labels.append(["Desired force", "Measured force"])
        self._viewer(
            self.graph,
            """local nextUpdate=0
function sysCall_sensing()
    graph.handle()
    if sim.getSimulationTime()>=nextUpdate then
        graph.updateCurves();nextUpdate=sim.getSimulationTime()+0.1
    end
end""",
        )

    def update(self, desired, measured):
        for stream, value in zip(self.streams, (desired, measured)):
            self.sim.setGraphStreamValue(self.graph, stream, float(value))
