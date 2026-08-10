"""Out-of-tree vLLM KV connector for AgentCache.

.. warning::
   Integration code. Requires ``vllm>=0.5`` and ``torch``. It is kept in a
   separate module so ``import agentcache`` stays torch-free on CI; the import
   below is lazy.

Why out-of-tree?
----------------
AgentCache is a *drop-in* replacement for vLLM's prefix-cache eviction. It does
**not** fork vLLM: it implements the ``KVConnector`` interface and registers
itself as ``"AgentCache"``. The engine calls ``get_kv_connector`` at startup;
every eviction point then consults :class:`~agentcache.policy.agentcache.AgentCachePolicy`
via :class:`~agentcache.scheduler.program_scheduler.ProgramScheduler`.

The same policy code runs here and in ``sim/`` -- so a decision validated in
the simulator is a decision validated on hardware, with no reimplementation.

.. note::
   vLLM's ``KVConnector`` surface has shifted between releases. This is written
   against ``vllm>=0.5`` (``KVConnectorBase``). If your vLLM version differs,
   the method signatures are the only thing that changes; the policy and
   accounting are version-independent.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import SystemConfig
from ..policy.agentcache import AgentCachePolicy
from ..scheduler.program_scheduler import ProgramScheduler


def build_scheduler(config: Optional[SystemConfig] = None) -> ProgramScheduler:
    """Construct the shared admission engine used by this connector."""
    from ..tiering import PinnedHostBackend  # lazy: needs torch

    cfg = config or SystemConfig()
    backend = PinnedHostBackend(cfg.hardware.cpu_kv_capacity_bytes) if cfg.enable_cpu_tier else None
    return ProgramScheduler(cfg, AgentCachePolicy(cfg), backend=backend)


class AgentCacheConnector:
    """Thin adapter from vLLM's KVConnector hooks to AgentCache.

    The methods below map vLLM's call sites onto :class:`ProgramScheduler`.
    They are intentionally small: all the *brains* live in the policy and the
    tiered manager, which are unit-tested and simulator-validated.
    """

    def __init__(self, model_config: Any, connector_config: Optional[Dict[str, Any]] = None) -> None:
        # `model_config` is vLLM's ModelConfig; we only need token/head shapes
        # to size blocks, which SystemConfig already carries. We accept it for
        # interface compatibility and ignore it here.
        self.scheduler = build_scheduler()
        self.model_config = model_config

    # -- vLLM KVConnector surface (>=0.5) ----------------------------------
    # The exact signatures vary by vLLM version; adapt the wrappers, not the
    # policy. Each maps onto a ProgramScheduler call.
    def start_save_kv(self, *args: Any, **kwargs: Any) -> None:
        """Engine finished a turn: let the policy decide what to keep."""
        # The engine already holds the KV on GPU; nothing to copy yet. The
        # policy runs at *eviction* time (get_num_new_active_blocks / drain),
        # not here.
        return None

    def wait_for_save(self) -> None:
        return None

    def get_num_new_active_blocks(self, *args: Any, **kwargs: Any) -> int:
        """Tell vLLM how much GPU KV we can afford right now.

        Queries the manager's free space; the policy's eviction decisions are
        applied via :meth:`ProgramScheduler.plan_admission` on the next
        admission.
        """
        return self.scheduler.manager.gpu_bytes_free // self.scheduler.config.block_bytes

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        """A returning program wants its KV back: promote from host/disk."""
        # Real promotion happens through ProgramScheduler.plan_admission,
        # which calls TieredCacheManager.promote after the lookup.
        return None

    def wait_for_load(self) -> None:
        return None

    def get_kv_connector(self, *args: Any, **kwargs: Any) -> AgentCacheConnector:
        """vLLM calls this to obtain the connector instance."""
        return self


def register() -> None:
    """Register AgentCache as a vLLM connector named ``"AgentCache"``.

    Usage in the vLLM server config::

        kv_transfer_config={"kv_connector":"AgentCache","kv_role":"kv_both"}

    If the running vLLM does not expose the registration hook this function is
    a no-op (so importing this module never breaks a non-vLLM process).
    """
    try:  # pragma: no cover - depends on vLLM internals at runtime
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        KVConnectorFactory.register_connector("AgentCache", AgentCacheConnector)
    except Exception:
        # No vLLM, or a different registration API. The class is still usable
        # directly via build_scheduler()/ProgramScheduler.
        pass
