"""Local data analysis agent. It runs one fixed pipeline every time."""

from agent.config import PIPELINE_VERSION, AgentConfig
from agent.guardrails import RefusalError
from agent.pipeline import run_pipeline

__version__ = PIPELINE_VERSION

__all__ = ["AgentConfig", "PIPELINE_VERSION", "RefusalError", "run_pipeline"]
