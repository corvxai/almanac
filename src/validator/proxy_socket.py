"""UNIX domain socket permissions for the validator-local signing proxy.

Agent sandboxes run as a fixed unprivileged uid (10001) and connect to the
proxy UDS through a read-only bind mount. The proxy process typically runs as
a different uid, so the socket must be connectable across uids. Access control
is enforced by ``X-Run-Id``, not socket ACLs.
"""

from __future__ import annotations

import os
from pathlib import Path

# World read/write — connect() requires write on the socket inode. Gated by
# X-Run-Id; per-run mount isolation prevents cross-run id discovery.
_PROXY_SOCKET_MODE = 0o666


def proxy_socket_bind_umask() -> int:
    """Widen the process umask so uvicorn creates a connectable UDS.

    Returns the previous umask so callers can restore it after bind.
    """
    return os.umask(0o000)


def make_proxy_socket_connectable(socket_path: Path) -> None:
    """Ensure ``socket_path`` is connectable by the sandbox uid."""
    os.chmod(socket_path, _PROXY_SOCKET_MODE)
