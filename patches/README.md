# vLLM integration patch

`vllm-da8ec282.patch` targets exactly
`da8ec2826898d47bd2fb424bc607052a6dc515c1` and keeps the controller outside
the CUDA hot path. Install this repository into the same Python environment
before applying the patch.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout da8ec2826898d47bd2fb424bc607052a6dc515c1
git apply --check ../specpilot-vllm/patches/vllm-da8ec282.patch
git apply ../specpilot-vllm/patches/vllm-da8ec282.patch
pip install -e ../specpilot-vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
```

The patch is a prototype integration seam, not an upstreamed contribution. It
does four narrowly scoped things:

1. validates the `dynamic_depth` configuration;
2. loads a measured profile during scheduler initialization;
3. chooses one batch-uniform K before each round;
4. feeds already-host-side accepted/proposed counts back after the round.

It does not call `.item()`, `.tolist()`, `cudaDeviceSynchronize`, or write files
in the decode loop. `shadow_goodput` runs the same estimator while keeping the
active K fixed at `static_k`.

