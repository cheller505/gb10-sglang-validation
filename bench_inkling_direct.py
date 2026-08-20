#!/usr/bin/env python3
"""Benchmark inkling-small directly on larry (bypass tunnel/proxy).
Measures TTFT, tokens/s, and inter-token latency via streaming.
Gentler test: shorter outputs, cooldown between requests."""
import time, json, requests, statistics, sys

BASE = "http://<LARRY_HOST>:30000/v1"
MODEL = "inkling-small"
HEADERS = {"Content-Type": "application/json"}

def benchmark(prompt, max_tokens, label):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_start = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, stream=True, timeout=300)
        r.raise_for_status()
    except Exception as e:
        print(f"\n  {label}: FAILED - {e}")
        return None

    ttft = None
    token_times = []
    content = ""
    total_tokens = 0
    prompt_tokens = 0

    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            total_tokens = u.get("completion_tokens", 0)
            prompt_tokens = u.get("prompt_tokens", 0)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {}).get("content")
            if delta:
                now = time.time()
                if ttft is None:
                    ttft = now - t_start
                token_times.append(now)
                content += delta

    t_end = time.time()
    total_time = t_end - t_start
    gen_time = (t_end - token_times[0]) if token_times else 0
    tps = total_tokens / gen_time if gen_time > 0 else 0
    intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times)) if token_times[i] - token_times[i-1] > 0]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Prompt:          {prompt_tokens} tokens")
    print(f"  Completion:      {total_tokens} tokens")
    print(f"  TTFT:            {ttft*1000:.0f} ms" if ttft else "  TTFT:            N/A")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Gen time:        {gen_time:.2f}s")
    print(f"  Tokens/s:        {tps:.1f}")
    if intervals:
        print(f"  Median interval: {statistics.median(intervals)*1000:.1f} ms")
        print(f"  Mean interval:   {statistics.mean(intervals)*1000:.1f} ms")
        print(f"  P95 interval:    {sorted(intervals)[int(len(intervals)*0.95)]*1000:.1f} ms")
    print(f"  Output preview:  {content[:150]!r}")
    return {"label": label, "prompt_tokens": prompt_tokens, "completion_tokens": total_tokens,
            "ttft_ms": round(ttft*1000) if ttft else None, "total_time": round(total_time, 2),
            "tokens_per_s": round(tps, 1)}

print("### Inkling-Small Performance Benchmark (direct on larry) ###")
print(f"### Endpoint: {BASE} ###")

results = []

# Test 1: short prompt, short output (latency)
results.append(benchmark(
    "What is the capital of France? One word answer.",
    32,
    "Short prompt, short output (latency)"
))

time.sleep(5)  # cooldown

# Test 2: short prompt, medium output
results.append(benchmark(
    "Write a paragraph about the ocean.",
    128,
    "Short prompt, medium output (128 tok)"
))

time.sleep(5)

# Test 3: short prompt, longer output (throughput stress)
results.append(benchmark(
    "Write a short essay about the history of computing, from Babbage to modern AI.",
    512,
    "Short prompt, long output (512 tok)"
))

time.sleep(5)

# Test 4: reasoning test
results.append(benchmark(
    "A farmer has 100 meters of fencing to enclose a rectangular field "
    "along a river (river side needs no fence). What dimensions maximize "
    "the area? Show your work.",
    512,
    "Reasoning test (math word problem)"
))

time.sleep(5)

# Test 5: verify model still alive after all tests
results.append(benchmark(
    "Say hello.",
    16,
    "Post-benchmark health check"
))

# Summary
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  {'Test':<45s} {'Prompt':>6s} {'Output':>7s} {'TTFT':>6s} {'Tok/s':>6s} {'Total':>6s}")
print(f"  {'':45s} {'tok':>6s} {'tok':>7s} {'ms':>6s} {'':>6s} {'s':>6s}")
print(f"  {'-'*82}")
for r in results:
    if r is None:
        continue
    ttft = f"{r['ttft_ms']}" if r['ttft_ms'] else "N/A"
    print(f"  {r['label']:<45s} {r['prompt_tokens']:>6d} {r['completion_tokens']:>7d} {ttft:>6s} {r['tokens_per_s']:>6.1f} {r['total_time']:>6.2f}")

alive = results[-1] is not None
print(f"\n  Model alive after benchmark: {alive}")
