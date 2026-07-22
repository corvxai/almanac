from __future__ import annotations

import pytest

from src.agent.loader import load_agent_class_from_code


def test_loads_single_agent_and_ignores_imported_agent_classes() -> None:
    code = """
from src.agent.base import BaseAgent
from src.agent.examples.v1_agent1_llm_only import V1Agent1LLMOnly

class MyAgent(BaseAgent):
    pass
"""

    loaded = load_agent_class_from_code(code)

    assert loaded.__name__ == "MyAgent"


def test_rejects_multiple_locally_defined_agents() -> None:
    code = """
from src.agent.base import BaseAgent

class FirstAgent(BaseAgent):
    pass

class SecondAgent(BaseAgent):
    pass
"""

    with pytest.raises(RuntimeError, match="multiple BaseAgent subclasses"):
        load_agent_class_from_code(code)
