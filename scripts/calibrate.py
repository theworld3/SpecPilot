"""Calibrate AgentCache's latency constants to a real GPU.

The simulator and scheduler take every latency number from
:class:`~agentcache.config.HardwareConfig`. Those defaults are a 4090; on
other hardware they should be measured, not guessed. This script writes a JSON
override that ``load_config()`` consumes::

    {"hardware": {"gpu_kv_capacity_bytes": ..., "h2d_bandwidth_bps": ...},
     "max_batch_size": 8}

Usage::

    # Measure PCIe bandwidths (needs torch + CUDA), keep prefill/decode at
    # values you obtained from a real engine profile:
    python scripts/calibrate.py --out agentcache_calibrated.json \\
        --prefill-tps 11000 --decode-step 0.028 --gpu-kv-gib 14

    # Or just emit a template to fill in by hand (no GPU required):
    python scripts/calibrate.py --template-only
"""
from __future__ import annotations

import argparse
import json

GIB = 1024 ** 3


def _measure_pci_bandwidth(device: str = "cuda") -> tuple[float, float]:
    """Measure host<->device bandwidth with pinned memory, in bytes/s."""
    import torch

    nbytes = 256 * 1024 ** 2  # 256 MiB
    host = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True)
    dev = torch.empty(nbytes // 4, dtype=torch.float32, device=device)
    n = 20

    # device -> host
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        host.copy_(dev, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    d2h = (nbytes * n) / (start.elapsed_time(end) / 1000.0)

    # host -> device
    start.record()
    for _ in range(n):
        dev.copy_(host, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    h2d = (nbytes * n) / (start.elapsed_time(end) / 1000.0)
    return h2d, d2h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="agentcache_calibrated.json")
    ap.add_argument("--prefill-tps", type=float, default=11000.0,
                   help="prefill tokens/s from a real engine profile")
    ap.add_argument("--decode-step", type=float, default=0.028,
                   help="per-request per-token decode latency (s)")
    ap.add_argument("--gpu-kv-gib", type=float, default=14.0,
                   help="KV pool size after weights/activations")
    ap.add_argument("--cpu-kv-gib", type=float, default=32.0)
    ap.add_argument("--template-only", action="store_true",
                   help="emit a hand-fillable template, no measurement")
    args = ap.parse_args()

    override: dict = {
        "hardware": {
            "gpu_kv_capacity_bytes": int(args.gpu_kv_gib * GIB),
            "cpu_kv_capacity_bytes": int(args.cpu_kv_gib * GIB),
            "prefill_tokens_per_s": args.prefill_tps,
            "decode_step_s": args.decode_step,
        }
    }

    if not args.template_only:
        try:
            import torch

            if torch.cuda.is_available():
                h2d, d2h = _measure_pci_bandwidth()
                override["hardware"]["h2d_bandwidth_bps"] = int(h2d)
                override["hardware"]["d2h_bandwidth_bps"] = int(d2h)
                print(f"measured: h2d={h2d/ GIB:.2f} GiB/s, d2h={d2h/ GIB:.2f} GiB/s")
            else:
                print("WARNING: CUDA not available; leaving bandwidth defaults in place.")
        except Exception as e:  # pragma: no cover
            print(f"WARNING: could not measure bandwidth ({e}); using defaults.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(override, f, indent=2)
    print(f"wrote calibration override -> {args.out}")


if __name__ == "__main__":
    main()
