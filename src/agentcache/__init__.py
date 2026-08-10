"""AgentCache -- agent-aware KV cache retention for multi-turn LLM serving.

The core (config, types, policies, tiering, simulation) is pure stdlib and
imports on any machine. GPU pieces (Triton kernels, the vLLM connector) are
optional extras and are imported lazily.
"""

from .config import HardwareConfig, SystemConfig, load_config
from .policy import (
    AgentCachePolicy,
    EvictionPolicy,
    LRUOffloadPolicy,
    LRUPolicy,
    OraclePolicy,
    RetentionCostModel,
)
from .tiering import TieredCacheManager
from .types import AgentProgram, BlockMeta, ModelSpec, ProgramState, Tier, Turn

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SystemConfig",
    "HardwareConfig",
    "load_config",
    "ModelSpec",
    "Tier",
    "ProgramState",
    "BlockMeta",
    "Turn",
    "AgentProgram",
    "EvictionPolicy",
    "LRUPolicy",
    "LRUOffloadPolicy",
    "AgentCachePolicy",
    "OraclePolicy",
    "RetentionCostModel",
    "TieredCacheManager",
]
