# Validator container image.
#
# This is the *trusted* side of the trust boundary. Holds the orchestrator,
# the validator-local signing proxy, and the Bittensor wallet (mounted
# read-only at runtime). It also has access to /var/run/docker.sock so it
# can spawn agent-runner sibling containers.
#
# The agent's untrusted code never executes inside this image — see
# `docker/agent-runner.Dockerfile`.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# System deps:
# - docker.io: CLI + libraries for ``docker wait`` (subprocess) and the Python
#   SDK over the bind-mounted socket. Many slim/apt images default to skipping
#   recommended packages; ``-o APT::Install-Recommends=true`` ensures the
#   ``docker`` client is installed alongside ``docker.io`` on current Debian.
# - ca-certificates / curl: TLS to the central gateway.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ca-certificates curl \
 && apt-get install -y -o APT::Install-Recommends=true docker.io \
 && rm -rf /var/lib/apt/lists/* \
 && test -x /usr/bin/docker \
 && /usr/bin/docker --version

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# UDS socket dir for the local proxy. Compose mounts a host directory onto
# this path so the agent-runner sibling container can bind-mount the same
# socket file read-only.
RUN mkdir -p /var/run/arcratio

# Default command runs the Bittensor validator loop. It starts
# the local signing proxy + agent orchestrator on a daemon thread internally
# (single-process v1), then drives the per-epoch scoring + set_weights flow.
# For dev/ad-hoc agent runs without chain interaction, override with
# `python scripts/run_forecast.py`.
CMD ["python", "scripts/run_validator.py"]
