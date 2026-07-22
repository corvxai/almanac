"""Shared helpers for loading a single-file forecasting agent."""

from __future__ import annotations

from typing import Any

from src.agent.base import BaseAgent


def load_agent_class_from_code(
    code: str,
    class_name: str | None = None,
    *,
    filename: str = "<agent>",
) -> type[BaseAgent]:
    """Execute agent source and return its sole BaseAgent subclass."""
    scope: dict[str, Any] = {"__name__": "__almanac_agent__"}
    exec(compile(code, filename, "exec"), scope)  # noqa: S102
    if class_name:
        cls = scope.get(class_name)
        if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
            raise RuntimeError(f"class '{class_name}' is not a BaseAgent subclass")
        return cls

    candidates = [
        obj
        for obj in scope.values()
        if isinstance(obj, type)
        and issubclass(obj, BaseAgent)
        and obj is not BaseAgent
        and obj.__module__ == "__almanac_agent__"
    ]
    if not candidates:
        raise RuntimeError("no BaseAgent subclass found in agent file")
    if len(candidates) > 1:
        names = ", ".join(sorted(candidate.__name__ for candidate in candidates))
        raise RuntimeError(f"multiple BaseAgent subclasses found in agent file ({names})")
    return candidates[0]
