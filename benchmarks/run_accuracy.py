"""Measure the accuracy delta introduced by FP8 KV offload.

Integration / accuracy harness. Quantising KV to FP8 before the GPU->host copy
halves transfer time and host footprint; this script quantifies the cost.

On a box with ``torch`` it simulates a KV cache of a given shape, casts
FP16 -> FP8(e4m3) -> FP16, and reports the reconstruction error. This is a
direct, model-free measurement of the FP8 rounding error -- the dominant
accuracy term of ``fp8_offload=True``.

For a production decision, replace the synthetic tensor with a real model's KV
(load a small LM, capture its KV on a held-out prompt, quantize, and compare
downstream logits / perplexity). The math below is the same; only the source
of ``kv`` changes.

Usage::

    python benchmarks/run_accuracy.py --layers 32 --heads 8 --dim 128 --seq 2048
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402


def _requires_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    if not _requires_torch():
        print("SKIP: torch not installed. Install with the `[gpu]` extra.")
        sys.exit(0)

    import torch

    max_err, mean_err, worst_cos = 0.0, 0.0, 1.0
    for _ in range(args.trials):
        kv = torch.randn(args.layers, args.seq, args.heads, args.dim, dtype=torch.float16)
        q8 = kv.to(torch.float8_e4m3fn).to(torch.float16)
        err = (kv - q8).abs()
        max_err = max(max_err, err.max().item())
        mean_err += err.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            kv.reshape(-1), q8.reshape(-1), dim=0
        ).item()
        worst_cos = min(worst_cos, cos)

    mean_err /= args.trials
    print(f"FP8(e4m3) KV reconstruction over {args.trials} trials "
          f"[{args.layers}x{args.seq}x{args.heads}x{args.dim}]:")
    print(f"  mean abs error : {mean_err:.4f} (fp16 units)")
    print(f"  max  abs error : {max_err:.4f}")
    print(f"  worst cosine   : {worst_cos:.6f}")
    print("Interpretation: values near 0 mean error and ~1 cosine => the FP8 "
          "offload is effectively lossless for attention scoring at this scale.")


if __name__ == "__main__":
    main()
