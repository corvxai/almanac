# Agent sandbox runner image.
#
# Built once, run by the validator's `sandbox_docker.py` as a sibling
# container per agent execution. This image is the agent's untrusted side
# of the trust boundary:
#
#   - No `bittensor` / no chain libraries.
#   - No validator code, no gateway server, no storage code.
#   - Only the curated runner-requirements.txt deps.
#
# Network isolation, FS hardening, and resource limits are applied by the
# *runtime* (`docker run --network=none --read-only ...`) — see
# `src/validator/forecasting/sandbox_docker.py`. This Dockerfile only sets up the
# minimum software environment.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root runtime user. The runtime also sets --user=10001:10001 as defence
# in depth.
RUN groupadd --system --gid 10001 runner \
 && useradd  --system --uid 10001 --gid 10001 --create-home --home-dir /home/runner runner

WORKDIR /opt/runner

COPY docker/runner-requirements.txt /opt/runner/runner-requirements.txt
RUN pip install --no-cache-dir -r /opt/runner/runner-requirements.txt

# Curated source copy — only the modules the runner needs. Anything not
# listed here is unavailable to the agent inside the sandbox.
COPY src/agent                       /opt/runner/src/agent
COPY src/core                        /opt/runner/src/core
COPY src/gateway/__init__.py         /opt/runner/src/gateway/__init__.py
COPY src/gateway/client.py           /opt/runner/src/gateway/client.py
COPY src/gateway/constants.py        /opt/runner/src/gateway/constants.py
COPY src/gateway/provider_capabilities.py /opt/runner/src/gateway/provider_capabilities.py
COPY src/gateway/providers/__init__.py /opt/runner/src/gateway/providers/__init__.py
COPY src/gateway/providers/base.py   /opt/runner/src/gateway/providers/base.py
COPY src/__init__.py                 /opt/runner/src/__init__.py

ENV PYTHONPATH=/opt/runner

# The validator-local proxy mounts its UDS into /run/almanac/proxy.sock and
# sets SANDBOX_PROXY_URL + RUN_ID. The runtime overrides USER too.
USER 10001:10001

ENTRYPOINT ["python", "-m", "src.agent.runner_entrypoint"]
