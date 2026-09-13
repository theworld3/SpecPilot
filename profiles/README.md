# Cost profiles

Files named `local_*.json` are ignored because a profile is specific to one
GPU, model revision, vLLM commit, graph configuration, and context bucket.
Generate one with `benchmarks.profile_cost` and keep its completed provenance
under `results/manifests/`.

Synthetic profiles are CI fixtures only and must carry `metadata.synthetic=true`.

