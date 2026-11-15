"""Local, deterministic data analysis agent for tabular data.

The agent runs a fixed six-stage pipeline on a single CSV or Excel file:

    profile -> clean -> analyze -> visualize -> predict -> report

It uses no generative model and makes no network calls. Every explanation
in the report is filled in from statistics the pipeline actually computed,
and every run saves its configuration and transformation log so the result
can be reproduced exactly.
"""

from agent.config import PIPELINE_VERSION, AgentConfig
from agent.guardrails import RefusalError
from agent.pipeline import run_pipeline

__version__ = PIPELINE_VERSION

__all__ = ["AgentConfig", "PIPELINE_VERSION", "RefusalError", "run_pipeline"]
