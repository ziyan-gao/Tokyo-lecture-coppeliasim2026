# Python client only: run the native CoppeliaSim GUI on the host.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libgomp is needed by native numerical/geometry packages; git supports VS Code.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Install CPU PyTorch first to avoid downloading CUDA dependencies on x86 Linux.
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3' \
    && pip install -r /tmp/requirements.txt ipykernel

WORKDIR /work
CMD ["bash"]
