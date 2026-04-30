"""Adversarial agent examples used by `tests/security/test_sandbox_lockdown.py`.

Each module here implements a single attack vector. The agent is *expected*
to fail when run inside the hardened sandbox — the test asserts the failure
mode (non-zero exit, specific error, etc.). These should never be wired into
production.
"""
