#!/usr/bin/env python3
"""Benchmark all 3 moe models directly on moe (<MOE_HOST>).
Runs on moe itself, hits localhost to eliminate network."""
import base64, json, time, statistics, os, urllib.request, urllib.error
from PIL import Image, ImageDraw, ImageFont

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time()-t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time()-t0
    except Exception as e:
        return -1, str(e), time.time()-t0

def get(url, timeout=10):
    req = urllib.request.Request(url)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time()-t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time()-t0
    except Exception as e:
        return -1, str(e), time.time()-t0

def get_model_id(port):
    sc, body, _ = get(f"http://localhost:{port}/v1/models")
    return json.loads(body)["data"][0]["id"]

def post_stream(url, payload, timeout=300):
    """Streaming POST to measure per-token timing."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        return -1, str(e), 0
    ttft = None
    token_times = []
    content = ""
    total_tokens = 0
    prompt_tokens = 0
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "): continue
        data = line[6:]
        if data == "[DONE]": break
        try: chunk = json.loads(data)
        except: continue
        if chunk.get("usage"):
            total_tokens = chunk["usage"].get("completion_tokens", 0)
            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
        for ch in chunk.get("choices", []):
            delta = ch.get("delta", {}).get("content") if "delta" in ch else ch.get("message",{}).get("content")
            if delta:
                now = time.time()
                if ttft is None: ttft = now - t0
                token_times.append(now)
                content += delta
    t_end = time.time()
    return {"ttft_ms": round(ttft*1000) if ttft else None,
            "total_time": round(t_end-t0, 3),
            "gen_time": round(t_end-token_times[0], 3) if token_times else 0,
            "tokens": total_tokens, "prompt_tokens": prompt_tokens,
            "tps": round(total_tokens/(t_end-token_times[0]), 1) if token_times and (t_end-token_times[0])>0 else 0,
            "content": content}

# ---- OCR benchmark ----
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
def make_ocr_img(text, size=36, W=1200, H=160):
    img = Image.new("RGB", (W,H), "white"); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(SANS_B, size); bb = d.textbbox((0,0), text, font=f)
    w = bb[2]-bb[0]; h = bb[3]-bb[1]
    d.text(((W-w)/2-bb[0], (H-h)/2-bb[1]), text, fill="black", font=f)
    buf = __import__("io").BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def make_doc_img():
    W,H = 960,540; img = Image.new("RGB",(W,H),"#fafafa"); d = ImageDraw.Draw(img)
    d.rectangle([0,0,W,78], fill="#1f3a5f")
    ft = ImageFont.truetype(SANS_B, 42); t = "QUARTERLY REPORT"
    bb = d.textbbox((0,0),t,font=ft); d.text(((W-(bb[2]-bb[0]))/2-bb[0],16-bb[1]), t, fill="white", font=ft)
    cy = 96
    d.text((70,cy), "Fiscal Year 2026 - Q2 Summary", fill="#222222", font=ImageFont.truetype(SANS_B,26)); cy += 48
    d.line([(70,cy),(W-70,cy)], fill="#cccccc", width=2); cy += 24
    for line in ["Revenue increased by 17 percent year over year,",
                 "driven by strong demand in the embedded systems",
                 "market. Operating margins improved to 23.4 percent."]:
        d.text((70,cy), line, fill="#111111", font=ImageFont.truetype(SERIF,26))
        bb = d.textbbox((0,0),line,font=ImageFont.truetype(SERIF,26)); cy += (bb[3]-bb[1])+12
    cy += 28; d.text((70,cy), "Confidential - Internal Use Only", fill="#888888", font=ImageFont.truetype(SANS_B,18))
    buf = __import__("io").BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def bench_ocr():
    print("\n### DeepSeek-OCR-2 Benchmark (port 30000) ###")
    BASE = "http://localhost:30000/v1/chat/completions"
    MODEL = "deepseek-ocr2"  # will be overridden by served name
    # get actual model id
    mid = get_model_id(30000)

    results = []
    # Test 1: simple text image
    for label, img_url, expected in [
        ("Simple text (pangram)", make_ocr_img("The quick brown fox jumps over the lazy dog."), "The quick brown fox"),
        ("Document-style (multi-line)", make_doc_img(), "QUARTERLY REPORT"),
    ]:
        payload = {"model": mid, "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url": img_url}},
            {"type":"text","text":"Transcribe all text in this image."}
        ]}], "max_tokens": 512, "temperature": 0.0}
        t0 = time.time()
        sc, body, dt = post(BASE, payload, timeout=120)
        t1 = time.time()
        try:
            j = json.loads(body)
            out = j["choices"][0]["message"].get("content","") or ""
            u = j.get("usage",{})
            ok = expected.lower() in out.lower()
            print(f"\n  {label}")
            print(f"    HTTP {sc} | {dt:.2f}s | prompt_tok={u.get('prompt_tokens')} comp_tok={u.get('completion_tokens')} | match={ok}")
            print(f"    Output: {out[:100]!r}")
            results.append({"label": label, "time": dt, "prompt_tok": u.get('prompt_tokens'), "comp_tok": u.get('completion_tokens'), "match": ok})
        except:
            print(f"  {label}: ERROR {body[:200]}")

    # Test 2: latency x3 (same image, measure variance)
    img = make_ocr_img("Invoice #12345 - Total: $999.42")
    print("\n  Latency test (same image x3):")
    for i in range(3):
        payload = {"model": mid, "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url": img}},
            {"type":"text","text":"Transcribe the text."}
        ]}], "max_tokens": 64, "temperature": 0.0}
        t0 = time.time(); sc, body, dt = post(BASE, payload, timeout=60); t1 = time.time()
        try:
            j = json.loads(body); out = j["choices"][0]["message"].get("content","") or ""
            print(f"    Run {i+1}: {dt:.3f}s | {out!r}")
        except:
            print(f"    Run {i+1}: ERROR")
    return results

def bench_embedding():
    print("\n### Qwen3-Embedding-0.6B Benchmark (port 30001) ###")
    BASE = "http://localhost:30001/v1/embeddings"
    mid = get_model_id(30001)

    results = []
    # Test 1: single embedding latency
    print("\n  Single embedding latency (x5):")
    times = []
    for i in range(5):
        payload = {"model": mid, "input": "The cat sat on the mat."}
        t0 = time.time(); sc, body, dt = post(BASE, payload, timeout=30)
        times.append(dt)
        if i == 0:
            j = json.loads(body); dim = len(j["data"][0]["embedding"])
            print(f"    dim={dim} (first run {dt:.3f}s)")
        else:
            print(f"    Run {i+1}: {dt:.3f}s")
    print(f"    Median: {statistics.median(times)*1000:.1f}ms | Mean: {statistics.mean(times)*1000:.1f}ms | Min: {min(times)*1000:.1f}ms")

    # Test 2: batch embeddings (varying batch sizes)
    print("\n  Batch embedding throughput:")
    sentences = [f"This is test sentence number {i} for batch embedding benchmark." for i in range(64)]
    for bs in [1, 4, 8, 16, 32, 64]:
        payload = {"model": mid, "input": sentences[:bs]}
        t0 = time.time(); sc, body, dt = post(BASE, payload, timeout=60)
        try:
            j = json.loads(body); n = len(j["data"])
            tps = n / dt
            print(f"    batch={bs:>3d} | {dt:.3f}s | {tps:.1f} embed/s")
        except:
            print(f"    batch={bs:>3d} | ERROR {body[:100]}")

    # Test 3: long text
    long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 100)
    payload = {"model": mid, "input": long_text}
    t0 = time.time(); sc, body, dt = post(BASE, payload, timeout=30)
    try:
        j = json.loads(body); u = j.get("usage",{})
        print(f"\n  Long text ({len(long_text)} chars): {dt:.3f}s | prompt_tok={u.get('prompt_tokens')}")
    except:
        print(f"  Long text: ERROR")
    return results

def bench_reranker():
    print("\n### Qwen3-Reranker-0.6B Benchmark (port 30002) ###")
    BASE = "http://localhost:30002/v1/chat/completions"
    mid = get_model_id(30002)
    SYS = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
    QUERY = "How do I bake sourdough bread at home?"
    DOC_REL = "Sourdough baking begins with a wild-yeast starter made from flour and water. After feeding the starter for several days, you mix it with more flour, water, and salt, then allow a long bulk fermentation before shaping, proofing, and baking in a hot Dutch oven."
    DOC_IRREL = "The 2024 Summer Olympics were held in Paris, France, featuring 32 sports and 329 events across 48 disciplines."

    def fmt(q, d): return f"<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}"

    # Test 1: latency x5 (alternating relevant/irrelevant)
    print("\n  Latency test (x6, alternating relevant/irrelevant):")
    times = []
    for i in range(6):
        doc = DOC_REL if i % 2 == 0 else DOC_IRREL
        expected = "yes" if i % 2 == 0 else "no"
        payload = {"model": mid, "messages": [
            {"role":"system","content":SYS},
            {"role":"user","content":fmt(QUERY, doc)}
        ], "max_tokens": 8, "temperature": 0.0, "logprobs": True, "top_logprobs": 5}
        t0 = time.time(); sc, body, dt = post(BASE, payload, timeout=30)
        times.append(dt)
        try:
            j = json.loads(body)
            out = j["choices"][0]["message"].get("content","")
            u = j.get("usage",{})
            ok = out.strip().lower().startswith(expected)
            print(f"    Run {i+1} ({expected:3s}): {dt:.3f}s | out={out!r} | prompt_tok={u.get('prompt_tokens')} comp_tok={u.get('completion_tokens')} | correct={ok}")
        except:
            print(f"    Run {i+1}: ERROR {body[:100]}")
    print(f"    Median: {statistics.median(times)*1000:.1f}ms | Mean: {statistics.mean(times)*1000:.1f}ms")

    # Test 2: with logprobs (graded scoring)
    print("\n  Logprob scoring (relevant vs irrelevant):")
    import math
    for label, doc in [("RELEVANT", DOC_REL), ("IRRELEVANT", DOC_IRREL)]:
        payload = {"model": mid, "messages": [
            {"role":"system","content":SYS},
            {"role":"user","content":fmt(QUERY, doc)}
        ], "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 10}
        sc, body, dt = post(BASE, payload, timeout=30)
        try:
            j = json.loads(body)
            ch = j["choices"][0]
            lps = (ch.get("logprobs") or {}).get("content") or []
            yes_lp = None
            if lps:
                for t in (lps[0].get("top_logprobs") or []):
                    if t.get("token","").lower().lstrip().startswith("yes"):
                        yes_lp = t.get("logprob")
            p_yes = math.exp(yes_lp) if yes_lp else 0
            print(f"    {label:12s}: P(yes)={p_yes:.4f} | {dt:.3f}s")
        except:
            print(f"    {label}: ERROR")

print("="*60)
print("  MOE MODEL BENCHMARKS")
print("="*60)
bench_ocr()
print()
bench_embedding()
print()
bench_reranker()
print("\n" + "="*60)
print("  DONE")
print("="*60)
