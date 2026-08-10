"""vLLM integration (requires vllm + torch).

Importing this package does not import torch; only :func:`build_scheduler`
and :class:`AgentCacheConnector` pull it in, lazily.
"""
from .kv_connector import AgentCacheConnector, build_scheduler, register

__all__ = ["AgentCacheConnector", "build_scheduler", "register"]
