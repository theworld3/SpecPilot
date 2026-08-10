"""Tiered KV storage: metadata management + pluggable byte movers."""

from .backend import NullBackend, StorageBackend
from .manager import LookupResult, TieredCacheManager

__all__ = ["TieredCacheManager", "LookupResult", "StorageBackend", "NullBackend"]


def __getattr__(name: str):
    # PinnedHostBackend needs torch; import lazily so `import agentcache`
    # stays torch-free on CI.
    if name == "PinnedHostBackend":
        from .backend import PinnedHostBackend

        return PinnedHostBackend
    raise AttributeError(name)
