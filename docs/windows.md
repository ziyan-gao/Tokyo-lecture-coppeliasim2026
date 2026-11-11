# Windows setup

Use **Windows 11 (x86-64)** with hardware virtualization enabled. CoppeliaSim runs as a Windows application; the Python client runs in Docker. Run the course commands below from **PowerShell**.

## 1. Enable WSL 2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart Windows when prompted. Then update WSL and check its version:

```powershell
wsl --update
wsl --version
```

See Microsoft's [WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install) if installation fails.

## 2. Install Docker Desktop

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and start it.

- Use the **WSL 2 backend**. Under **Settings > General**, enable **Use the WSL 2 based engine** if the option is shown; it may already be enabled by default.
- Use **Linux containers**. If Docker's tray menu offers **Switch to Linux containers**, select it.
- Wait until Docker Desktop reports that its engine is running.

Open a new, normal PowerShell window and verify:

```powershell
docker version
docker compose version
docker info --format '{{.OSType}}'
```

`docker version` should show both **Client** and **Server**. The last command should print `linux`.

## 3. Install Git and CoppeliaSim

- Install [Git for Windows](https://git-scm.com/download/win). Open a new PowerShell window afterwards so that `git` is available.
- Install the **Windows version of CoppeliaSim 4.10** from [Coppelia Robotics](https://www.coppeliarobotics.com/).
- Launch CoppeliaSim in Windows, create a new scene and leave the simulation **stopped**. Python will start and step it.

Python and the course dependencies are installed inside Docker; the build command below prepares them.

## 4. Download and build the course environment

In a normal PowerShell window:

```powershell
cd "$env:USERPROFILE"
git clone https://github.com/ziyan-gao/Tokyo-lecture-coppeliasim2026.git
cd Tokyo-lecture-coppeliasim2026
docker compose build
```

Keep Docker Desktop running. The first build downloads dependencies and can take several minutes.

## 5. Connect and run

Keep the CoppeliaSim window open. In the same repository folder:

```powershell
docker compose run --rm lab python -c "from shared.common import connect; connect()"
docker compose run --rm lab python lecture1/zmq_stepping.py
```

The first command should print `connected` and the CoppeliaSim version. In the second demonstration, press **Enter** to advance the falling ball by one simulation step; press **Ctrl+C** to finish.

For another demo:

```powershell
docker compose run --rm lab python lecture2/vision_sensor.py
```

Generated images and reports are saved in `scripts\out\` in your Windows checkout.

## Troubleshooting

| Symptom | Check |
|---|---|
| `docker` or `git` is not recognized | Finish installing the application, then open a new PowerShell window. |
| Docker cannot connect to its server | Start Docker Desktop and wait for its engine. |
| WSL reports a virtualization error | Enable hardware virtualization in BIOS/UEFI, then restart Windows. |
| CoppeliaSim connection times out | Keep CoppeliaSim open, check its ZMQ remote API server on port `23000`, and allow the connection through Windows Firewall on your private network. |
| The client tries to connect to `localhost` | Use the provided Compose command. It sets `COPPELIASIM_HOST=host.docker.internal`, which reaches the Windows host from the container. |

For VS Code, install the **Dev Containers** extension, open the cloned repository folder and run **Dev Containers: Reopen in Container**. CoppeliaSim remains open in Windows. See [Docker's WSL 2 guide](https://docs.docker.com/desktop/features/wsl/) for additional Docker settings.
