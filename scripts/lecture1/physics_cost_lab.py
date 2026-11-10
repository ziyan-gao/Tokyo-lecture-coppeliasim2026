"""Interactive multi-body physics cost lab (CoppeliaSim >= 4.8, Bullet).

    docker compose run --rm lab python lecture1/physics_cost_lab.py
    python scripts/lecture1/physics_cost_lab.py --self-test

Use the simulator panel to choose settings, then tick Run or Compare.
Compare rebuilds the same seeded scene, changing just the selected configuration
field. Each case is repeated; settings are restored after each trial.
Exports: scripts/out/physics_cost_lab/{timestamp}/results.json, summary.csv,
samples.csv, comparison.png. No robot, sensors, video, or external UI framework.
See physics_cost_lab_guide.txt for interpretation and remaining confounders.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

import numpy as np

from shared.common import OUT_DIR, clear_scene, connect, stop_simulation


@dataclass(frozen=True)
class Config:
    count: int = 60
    geometry: str = 'primitive'
    layout: str = 'pile'
    resolution: int = 12
    friction: float = 0.8
    iterations: int = 50
    main_dt: float = 0.05
    physics_dt: float = 0.005
    mass_ratio: float = 1.0
    collisions: bool = True
    visible: bool = True
    dynamic: bool = True
    seconds: float = 4.0
    repeats: int = 3


# All vertices are in the same frame as the compound analytic dumbbell.
def sphere_mesh(resolution, center):
    rings, sides = resolution, resolution * 2
    vertices = [[center[0], center[1], center[2] + 0.025]]
    for i in range(1, rings):
        theta = math.pi * i / rings
        for j in range(sides):
            phi = 2 * math.pi * j / sides
            vertices.append([center[0] + .025 * math.sin(theta) * math.cos(phi),
                             center[1] + .025 * math.sin(theta) * math.sin(phi),
                             center[2] + .025 * math.cos(theta)])
    bottom = len(vertices)
    vertices.append([center[0], center[1], center[2] - .025])
    faces = []
    for j in range(sides):
        a, b = 1 + j, 1 + (j + 1) % sides
        faces.append([0, a, b])
    for i in range(rings - 2):
        for j in range(sides):
            a = 1 + i * sides + j
            b = 1 + i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            faces.extend([[a, c, b], [b, c, d]])
    start = 1 + (rings - 2) * sides
    for j in range(sides):
        faces.append([start + j, bottom, start + (j + 1) % sides])
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def positions(config):
    """Non-overlapping starts; deterministic layout independent of rendering settings."""
    cols = max(1, math.ceil(math.sqrt(config.count)))
    for k in range(config.count):
        if config.layout == 'pile':
            yield [((k % 5) - 2) * .15, (((k // 5) % 4) - 1.5) * .15,
                   .15 + .15 * (k // 20)]
        elif config.layout == 'tower':
            yield [0, 0, .15 + .075 * k]
        else:
            yield [((k % cols) - (cols - 1) / 2) * .3,
                   ((k // cols) - (cols - 1) / 2) * .3, .15]


def template(sim, cfg):
    """Same three components; rendering complexity and collision geometry separated."""
    parts = []
    for x in (-.04, .04):
        h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [.05] * 3, 0)
        sim.setObjectPosition(h, [x, 0, 0], sim.handle_world)
        parts.append(h)
    rod = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [.08, .016, .016], 0)
    parts.append(rod)
    body = sim.groupShapes(parts, False)
    # A fixed rod mesh plus adjustable sphere tessellation gives deterministic counts.
    # Generate the mesh from sphere meshes and a small cuboid, not from render settings.
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * [.04,.008,.008]
    cube_faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                           [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    vertices, faces = [corners], [cube_faces]
    offset = 8
    for x in (-.04, .04):
        v, f = sphere_mesh(cfg.resolution, [x,0,0])
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    mesh = sim.createShape(0, .5, np.vstack(vertices).ravel().tolist(),
                           np.vstack(faces).ravel().tolist())
    triangle_count = sum(len(f) for f in faces)
    if cfg.geometry == 'mesh':
        sim.removeObjects([body])
        body = mesh
    else:
        sim.setBoolProperty(mesh, 'dynamic', False)
        sim.setBoolProperty(mesh, 'respondable', False)
        sim.setObjectParent(mesh, body, True)
        # Both cases render the same mesh; only collision representation changes.
        sim.setIntProperty(body, 'layer', 1 << 8)
    sim.computeMassAndInertia(body, 1000.)
    sim.setFloatProperty(body, 'mass', .15)
    sim.setBoolProperty(body, 'dynamic', cfg.dynamic)
    sim.setBoolProperty(body, 'respondable', cfg.collisions)
    return body, triangle_count


def build(sim, cfg):
    clear_scene(sim)
    root = sim.createDummy(.001)
    sim.setObjectAlias(root, 'Lab')
    extent = 6.0  # fixed across count comparisons; fits all supported sparse layouts
    floor = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [extent, extent, .04], 0)
    sim.setObjectPosition(floor, [0,0,-.02], sim.handle_world)
    sim.setBoolProperty(floor, 'dynamic', False)
    sim.setBoolProperty(floor, 'respondable', True)
    sim.setObjectParent(floor, root, True)
    sim.setFloatProperty(floor, 'bullet.friction', cfg.friction)
    if cfg.count == 0:
        return [], 0
    seed, triangle_count = template(sim, cfg)
    tree = sim.getObjectsInTree(seed)
    bodies = [seed]
    for _ in range(cfg.count - 1):
        bodies.append(sim.copyPasteObjects(tree, 0)[0])
    rng = np.random.default_rng(7)
    for k, (h, pos) in enumerate(zip(bodies, positions(cfg))):
        sim.setObjectPosition(h, pos, sim.handle_world)
        # Keep the tower initially aligned; pile/sparse share the same seeded rotations.
        orientation = [0,0,0] if cfg.layout == 'tower' else rng.uniform(-.3,.3,3).tolist()
        sim.setObjectOrientation(h, orientation, sim.handle_world)
        sim.setFloatProperty(h, 'mass', .15 * (cfg.mass_ratio if k % 2 else 1))
        sim.setFloatProperty(h, 'bullet.friction', cfg.friction)
        sim.setObjectParent(h, root, True)
        for shape in sim.getObjectsInTree(h, sim.object_shape_type, 0):
            if not cfg.visible:
                sim.setIntProperty(shape, 'layer', 0)
            sim.setShapeColor(shape, '', sim.colorcomponent_ambient_diffuse,
                              [.25,.65,.9] if cfg.geometry == 'primitive' else [.95,.5,.2])
    return bodies, triangle_count


@contextmanager
def parameters(sim, cfg):
    stop_simulation(sim)
    old_engine = sim.getInt32Param(sim.intparam_dynamic_engine)
    old_realtime = sim.getBoolProperty(sim.handle_scene, 'realtimeSimulation')
    old_dt = sim.getFloatProperty(sim.handle_scene, 'dynamicsStepSize')
    old_main_dt = sim.getFloatProperty(sim.handle_scene, 'timeStep')
    old_iterations = sim.getIntProperty(sim.handle_scene, 'bullet.iterations')
    try:
        sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_bullet)
        sim.setBoolProperty(sim.handle_scene, 'realtimeSimulation', False)
        sim.setFloatProperty(sim.handle_scene, 'timeStep', cfg.main_dt)
        sim.setFloatProperty(sim.handle_scene, 'dynamicsStepSize', cfg.physics_dt)
        sim.setIntProperty(sim.handle_scene, 'bullet.iterations', cfg.iterations)
        yield
    finally:
        stop_simulation(sim)
        sim.setStepping(False)
        sim.setInt32Param(sim.intparam_dynamic_engine, old_engine)
        sim.setFloatProperty(sim.handle_scene, 'timeStep', old_main_dt)
        sim.setFloatProperty(sim.handle_scene, 'dynamicsStepSize', old_dt)
        sim.setIntProperty(sim.handle_scene, 'bullet.iterations', old_iterations)
        sim.setBoolProperty(sim.handle_scene, 'realtimeSimulation', old_realtime)


def trial(sim, cfg, progress):
    with parameters(sim, cfg):
        start = time.perf_counter()
        bodies, triangles = build(sim, cfg)
        build_s = time.perf_counter() - start
        dt = sim.getSimulationTimeStep()
        actual_physics_dt = sim.getFloatProperty(sim.handle_scene, 'dynamicsStepSize')
        sim.setStepping(True)
        start = time.perf_counter()
        sim.startSimulation()
        startup_s = time.perf_counter() - start
        samples = []
        next_ui = time.perf_counter()
        for k in range(max(1, round(cfg.seconds / dt))):
            start = time.perf_counter()
            sim.step()
            samples.append(time.perf_counter() - start)
            if time.perf_counter() >= next_ui:
                progress(f'{k+1} steps | mean {np.mean(samples)*1000:.2f} ms | '
                         f'{(k+1)*dt:.2f}/{cfg.seconds:.2f} s')
                next_ui = time.perf_counter() + .4
        # These queries occur after timing; linear AND angular speed must be low.
        low_speed = sum(np.linalg.norm(v) < .02 and np.linalg.norm(w) < .1
                        for v, w in (sim.getObjectVelocity(h) for h in bodies))
        below = sum(sim.getObjectPosition(h, sim.handle_world)[2] < -.1 for h in bodies)
        return dict(config=asdict(cfg), build_s=build_s, startup_s=startup_s,
                    main_dt=dt, actual_physics_dt=actual_physics_dt,
                    triangles_per_visual=triangles, samples_s=samples,
                    mean_ms=float(np.mean(samples)*1000), p95_ms=float(np.percentile(samples,95)*1000),
                    first_ms=samples[0]*1000, tail_ms=float(np.mean(samples[len(samples)//2:])*1000),
                    step_only_rtf=len(samples)*dt/sum(samples),
                    low_speed=low_speed, below_floor=below)


FIELDS = [
    ('count', 'Object count (0 = empty baseline)', [0,20,60,120,240]),
    ('geometry', 'Collision representation', ['primitive','mesh']),
    ('layout', 'Spatial/contact arrangement', ['sparse','pile','tower']),
    ('resolution', 'Visual/mesh resolution', [6,12,24]),
    ('friction', 'Bullet friction (both surfaces)', [0.,.4,.8,1.2]),
    ('iterations', 'Bullet solver iterations', [10,50,100]),
    ('main_dt', 'Main simulation step (seconds)', [.025,.05,.1]),
    ('physics_dt', 'Physics substep (seconds)', [.01,.005,.0025]),
    ('mass_ratio', 'Alternating mass ratio', [1.,10.,100.]),
    ('collisions', 'Object collision response', [True,False]),
    ('visible', 'Object rendering', [True,False]),
    ('dynamic', 'Dynamic objects', [True,False]),
    ('seconds', 'Simulated duration', [2.,4.,8.]),
    ('repeats', 'Repeated trials', [1,3,5]),
]


class Panel:
    def __init__(self, api):
        self.api = api
        xml = ['<ui title="Physics cost lab | Bullet" closeable="false" resizable="true" layout="vbox">',
               '<label text="Select controls, then tick Run. Compare varies ONE selected field."/>',
               '<group layout="form">']
        for i, (_, label, values) in enumerate(FIELDS, 1):
            xml += [f'<label text="{label}"/>', f'<combobox id="{i}">']
            xml += [f'<item>{value}</item>' for value in values]
            xml += ['</combobox>']
        xml += ['<label text="Comparison factor"/><combobox id="30">']
        xml += [f'<item>{label}</item>' for _,label,_ in FIELDS[:-2]]
        xml += ['</combobox></group>', '<checkbox id="40" text="Run current settings"/>',
                '<checkbox id="41" text="Compare selected factor (all values)"/>',
                '<checkbox id="42" text="Exit after current experiment"/>',
                '<label id="50" text="Ready"/>', '<label id="51" text="No measurements yet"/>',
                '<label text="Step wall time includes simulator/render/remote overhead. Not pure physics time."/>',
                '<label text="Low-speed count is not an engine sleeping counter. No contact counts are inferred."/>',
                '</ui>']
        self.h = api.create(''.join(xml))
        defaults = asdict(Config())
        for i,(name,_,values) in enumerate(FIELDS,1):
            api.setComboboxSelectedIndex(self.h,i,values.index(defaults[name]),True)

    def config(self):
        return Config(**{name: values[self.api.getComboboxSelectedIndex(self.h,i)]
                         for i,(name,_,values) in enumerate(FIELDS,1)})

    def checked(self, widget):
        return self.api.getCheckboxValue(self.h, widget) != 0

    def status(self, text):
        self.api.setLabelText(self.h,50,text)

    def destroy(self):
        self.api.destroy(self.h)


def export(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(OUT_DIR) / 'physics_cost_lab' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True)
    metadata = dict(engine='bullet', measurement='sim.step wall time; not pure physics',
                    note='UI updates and diagnostic queries excluded; fixed seed 7; first step retained; not engine-internal profiling',
                    environment=dict(python=sys.version, numpy=np.__version__),
                    results=results)
    (folder/'results.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    keys = ['case','repeat',*asdict(Config()).keys(),'mean_ms','p95_ms','first_ms','tail_ms',
            'step_only_rtf','build_s','startup_s','low_speed','below_floor','actual_physics_dt','triangles_per_visual']
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k,row['config'].get(k)) for k in keys})
    with (folder/'samples.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['case','repeat','simulated_s','step_wall_ms'])
        for row in results:
            for i,value in enumerate(row['samples_s'],1):
                writer.writerow([row['case'],row['repeat'],i*row['main_dt'],value*1000])
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    labels=list(dict.fromkeys(r['case'] for r in results))
    groups=[[r['mean_ms'] for r in results if r['case']==label] for label in labels]
    axes[0].bar(range(len(labels)),[np.mean(g) for g in groups],
                yerr=[np.std(g) for g in groups],capsize=4)
    axes[0].set_xticks(range(len(labels)),labels,rotation=30,ha='right')
    axes[0].set_ylabel('Mean step wall time (ms); error bars = trial SD')
    for label in labels:
        row=next(r for r in results if r['case']==label)
        axes[1].plot(np.arange(1,len(row['samples_s'])+1)*row['main_dt'],
                     np.asarray(row['samples_s'])*1000,label=label)
    axes[1].set(xlabel='Simulated time (s), first repeat',ylabel='Step wall time (ms)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Includes simulator/render/remote overhead; compare cost AND stability')
    fig.tight_layout(); fig.savefig(folder/'comparison.png',dpi=140); plt.close(fig)
    return folder


def self_test():
    for resolution in (6,12,24):
        v,f=sphere_mesh(resolution,[.04,0,0])
        assert f.min()==0 and f.max()==len(v)-1
        edges={}
        for a,b,c in f:
            for x,y in ((a,b),(b,c),(c,a)):
                edge=tuple(sorted((int(x),int(y))))
                edges[edge]=edges.get(edge,0)+1
        assert all(n==2 for n in edges.values())
        assert len(v)-len(edges)+len(f)==2
        a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]]
        assert np.einsum('ij,ij->i',np.cross(b-a,c-a),(a+b+c)/3-[.04,0,0]).min()>0
    cfg=Config()
    assert list(positions(cfg))==list(positions(replace(cfg,visible=False)))
    assert len(list(positions(replace(cfg,count=240))))==240
    for name,_,values in FIELDS:
        assert getattr(cfg,name) in values
    print('PASS: closed outward mesh topology, deterministic positions, UI defaults')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    client,sim=connect()
    print(f'CoppeliaSim {sim.getInt32Param(sim.intparam_program_full_version)} | Bullet',flush=True)
    panel=Panel(client.require('simUI'))
    try:
        while not panel.checked(42):
            compare=panel.checked(41)
            if not (compare or panel.checked(40)):
                time.sleep(.1); continue
            panel.api.setCheckboxValue(panel.h,40,0,True)
            panel.api.setCheckboxValue(panel.h,41,0,True)
            cfg=panel.config()
            field=FIELDS[panel.api.getComboboxSelectedIndex(panel.h,30)]
            cases=[(f'{field[0]}={v}',replace(cfg,**{field[0]:v})) for v in field[2]] if compare else [('current',cfg)]
            # Alternate direction across repeats to reduce fixed-order thermal/cache bias.
            results=[]
            try:
                for repeat in range(cfg.repeats):
                    for label,case in cases[::1 if repeat%2==0 else -1]:
                        if panel.checked(42): break
                        panel.status(f'Building {label}, repeat {repeat+1}/{cfg.repeats}')
                        row=trial(sim,case,lambda text:panel.status(f'{label} | {text}'))
                        row.update(case=label,repeat=repeat+1)
                        results.append(row)
                        panel.api.setLabelText(panel.h,51,
                            f'{label}: mean {row["mean_ms"]:.2f}, P95 {row["p95_ms"]:.2f} ms | '
                            f'low-speed {row["low_speed"]}/{case.count} | below-floor {row["below_floor"]}')
                if results:
                    folder=export(results)
                    panel.status(f'Saved: {folder}')
                    print(f'Saved: {folder}',flush=True)
            except Exception as exc:
                panel.status(f'Experiment failed: {exc}')
                print(f'Experiment failed: {exc}',file=sys.stderr,flush=True)
    finally:
        panel.destroy()


if __name__=='__main__':
    main()
