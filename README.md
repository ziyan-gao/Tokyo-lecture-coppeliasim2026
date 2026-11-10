# CoppeliaSim Robotics Lectures 2026

Python demonstrations of robot kinematics, force control, RGB-D sensing, grasping and motion planning.

**CoppeliaSim runs on your computer; Docker runs the Python client.** They communicate through the ZeroMQ remote API on port `23000`.

## Install

Install **CoppeliaSim 4.10** for your OS from [Coppelia Robotics](https://www.coppeliarobotics.com/), plus Git and Docker Compose v2:

| System | Docker setup |
|---|---|
| Ubuntu 22.04 / 24.04, x86-64 | [Docker Engine + Compose plugin](https://docs.docker.com/engine/install/ubuntu/) |
| Windows 11, x86-64 | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/), **WSL 2**, **Linux containers** |
| macOS, Intel or Apple Silicon | [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/) matching your processor |

Use a supported desktop OS with a compatible CoppeliaSim build and graphics drivers. The Python container uses CPU PyTorch; GPU passthrough is unnecessary. Other systems/architectures are not guaranteed.

## Run

1. Clone this repository and build the environment:

   ```bash
   git clone https://github.com/ziyan-gao/Tokyo-lecture-coppeliasim2026.git
   cd Tokyo-lecture-coppeliasim2026
   docker compose build
   ```

   Start Docker before building. The first build downloads dependencies and may take several minutes.

2. Open CoppeliaSim, create a new scene and leave the simulation **stopped**.

3. Check the connection and run a demonstration:

   ```bash
   docker compose run --rm lab python -c "from shared.common import connect; connect()"
   docker compose run --rm lab python lecture1/zmq_stepping.py
   ```

   Press **Enter** to advance one step. Scripts start/step the simulation themselves. Stop the previous Python controller before running another demo.

More examples:

```bash
docker compose run --rm lab python lecture1/kuka_zmq_control.py
docker compose run --rm lab python lecture2/vision_sensor.py
docker compose run --rm lab python lecture2/rgbd_pointcloud.py
```

Lecture code is in `scripts/lecture1/` and `scripts/lecture2/`. Models and downloads are cached in `scripts/data/`; generated images/reports appear in `scripts/out/` on your computer.

**Optional VS Code:** install the **Dev Containers** extension, open this repository and select **Dev Containers: Reopen in Container**. CoppeliaSim stays on the host.

**Connection trouble?** Check CoppeliaSim's ZMQ server and firewall access to port `23000`. Inside Docker, use `host.docker.internal`, not `localhost`. The Ubuntu host-gateway mapping is already included in Compose.

Course code is MIT licensed. GR-ConvNet attribution is in `scripts/third_party/LICENSE.grconvnet`; YCB assets retain their original dataset licence.
