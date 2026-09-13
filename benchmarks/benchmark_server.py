"""OpenAI-compatible streaming benchmark for an already running vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    success: bool
    prompt_chars: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    error: str | None = None


async def _one_request(client, semaphore, args, request_id: int, prompt: str) -> RequestResult:
    started = time.perf_counter()
    first_token_at: float | None = None
    finished = started
    completion_tokens = 0
    try:
        async with semaphore:
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": args.ignore_eos,
            }
            async with client.stream("POST", "/v1/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    text = "".join(choice.get("text", "") for choice in chunk.get("choices", []))
                    now = time.perf_counter()
                    if text and first_token_at is None:
                        first_token_at = now
                    usage = chunk.get("usage")
                    if usage:
                        completion_tokens = int(usage.get("completion_tokens", 0))
                    finished = now
        if first_token_at is None:
            raise RuntimeError("stream completed without a token chunk")
        tpot = None
        if completion_tokens > 1:
            tpot = (finished - first_token_at) * 1000.0 / (completion_tokens - 1)
        return RequestResult(
            request_id=request_id,
            success=True,
            prompt_chars=len(prompt),
            completion_tokens=completion_tokens,
            latency_ms=(finished - started) * 1000.0,
            ttft_ms=(first_token_at - started) * 1000.0,
            tpot_ms=tpot,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            success=False,
            prompt_chars=len(prompt),
            completion_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ttft_ms=None,
            tpot_ms=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    return {
        name: float(np.percentile(values, percentile))
        for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99))
    }


def _parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(maxsplit=1)
            name = sample.split("{", 1)[0]
            totals[name] = totals.get(name, 0.0) + float(raw_value)
        except ValueError:
            continue
    return totals


async def _read_metrics(client) -> dict[str, float]:
    response = await client.get("/metrics")
    response.raise_for_status()
    return _parse_prometheus(response.text)


async def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("install the GPU extras: pip install -e '.[gpu]'") from exc

    prompts = []
    with Path(args.prompts).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                prompts.append(value["prompt"] if isinstance(value, dict) else str(value))
    if not prompts:
        raise ValueError("prompt JSONL is empty")
    selected = [prompts[index % len(prompts)] for index in range(args.requests)]
    semaphore = asyncio.Semaphore(args.concurrency)
    rng = np.random.default_rng(args.seed)

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        metrics_before = await _read_metrics(client) if args.collect_metrics else {}
        tasks = []
        wall_started = time.perf_counter()
        for request_id, prompt in enumerate(selected):
            tasks.append(
                asyncio.create_task(_one_request(client, semaphore, args, request_id, prompt))
            )
            if args.arrival_rate > 0 and request_id + 1 < len(selected):
                await asyncio.sleep(float(rng.exponential(1.0 / args.arrival_rate)))
        results = await asyncio.gather(*tasks)
        wall_seconds = time.perf_counter() - wall_started
        metrics_after = await _read_metrics(client) if args.collect_metrics else {}

    metric_deltas = {
        name: value - metrics_before.get(name, 0.0)
        for name, value in metrics_after.items()
        if name.startswith("vllm:spec_decode_")
    }

    successful = [result for result in results if result.success]
    tokens = sum(result.completion_tokens for result in successful)
    summary = {
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "request_throughput_rps": len(successful) / wall_seconds,
        "output_throughput_tps": tokens / wall_seconds,
        "ttft_ms": _percentiles(
            [result.ttft_ms for result in successful if result.ttft_ms is not None]
        ),
        "tpot_ms": _percentiles(
            [result.tpot_ms for result in successful if result.tpot_ms is not None]
        ),
        "latency_ms": _percentiles([result.latency_ms for result in successful]),
        "mean_completion_tokens": statistics.fmean(
            result.completion_tokens for result in successful
        )
        if successful
        else 0.0,
    }
    return {
        "schema_version": 1,
        "synthetic": False,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "arrival_rate": args.arrival_rate,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "seed": args.seed,
            "label": args.label,
            "k": args.k,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": summary,
        "spec_decode_metric_deltas": metric_deltas,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help='JSONL: {"prompt": "..."}')
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True, help="for example fixed_k3 or specpilot")
    parser.add_argument("--k", type=int, help="static K; omit for a dynamic policy")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--collect-metrics", action="store_true")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        raise ValueError("requests, concurrency, and max_tokens must be positive")
    payload = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    if payload["summary"]["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
