"""Fused gather + FP8 pack for KV offload.

.. warning::
   Integration code. Requires ``triton`` and a CUDA GPU to *run*. The module
   imports on a CPU-only box (triton is imported lazily) and falls back to a
   pure-torch implementation via :func:`pack_fp8_torch`.

Why fused?
----------
When a block is demoted GPU -> host we want to (a) select the blocks worth
keeping and (b) halve their footprint by casting FP16 -> FP8. Doing both in a
single kernel avoids a separate gather pass and a separate cast, and keeps the
hot KV tensor coherent in GPU memory instead of staging through a temp buffer.

The kernel is intentionally simple: one 1-D grid over output elements, each
thread maps its output element back to ``src[block][within]`` and casts. That
is enough to be a real, measurable win over two separate launches and is easy
to verify against the torch fallback in tests.
"""
from __future__ import annotations

from typing import Tuple

try:
    import torch
except Exception:  # pragma: no cover - CPU / no torch
    torch = None  # type: ignore

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU / no triton
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gather_pack_kernel(
        src_ptr,
        idx_ptr,
        dst_ptr,
        block_elems: tl.constexpr,
        n_out: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        # map output element -> (block position, within-block offset)
        pos = offs // block_elems
        within = offs % block_elems
        src_block = tl.load(idx_ptr + pos)
        src_idx = src_block * block_elems + within
        val = tl.load(src_ptr + src_idx, mask=mask)
        # FP16 -> FP8 (e4m3). Single rounding step.
        packed = val.to(tl.float8e4m3fn)
        tl.store(dst_ptr + offs, packed, mask=mask)


def pack_fp8(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``src[indices]`` and cast to FP8 (e4m3) in one kernel.

    ``src`` is ``[num_blocks, block_tokens, n_heads, head_dim]`` (FP16/FP32).
    ``indices`` selects which blocks to keep. Returns a contiguous FP8 tensor.
    """
    if not _HAS_TRITON or not src.is_cuda:
        return pack_fp8_torch(src, indices)

    num_blocks_sel = indices.numel()
    block_elems = src[0].numel()
    out_shape = (num_blocks_sel, *src.shape[1:])
    n_out = num_blocks_sel * block_elems
    dst = torch.empty(n_out, dtype=torch.float8_e4m3fn, device=src.device)

    BLOCK = 1024
    grid = (triton.cdiv(n_out, BLOCK),)
    _gather_pack_kernel[grid](
        src, indices, dst, block_elems, n_out, BLOCK
    )
    return dst.reshape(out_shape)


def pack_fp8_torch(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reference implementation: gather then cast. Same result, two launches."""
    return src[indices].to(torch.float8_e4m3fn)


def unpack_fp8(dst: torch.Tensor, src_shape: Tuple[int, ...]) -> torch.Tensor:
    """Inverse of :func:`pack_fp8` (FP8 -> FP16). Used on reload."""
    return dst.reshape(-1).to(torch.float16).reshape(src_shape)
