#!/usr/bin/env python3
"""Shakespeare OCR Pipeline — processes all pages of the complete works.
OCR (DeepSeek-OCR-2) → Embed (Qwen3-Embedding) → Index → Query (Retrieve + Rerank)
All via the Shadow LiteLLM API."""
import base64, json, math, time, os, sys, pickle, glob, io
import requests
from PIL import Image

BASE = "http://192.168.200.135:30000/v1"  # Direct to moe (bypass Shadow/LiteLLM for speed)
EMB_BASE = "http://192.168.200.135:30001/v1"
RERANK_BASE = "http://192.168.200.135:30002/v1"
KEY = ""
HEADERS = {"Content-Type": "application/json"}
PAGES_DIR = "/tmp/opencode/shakespeare/pages"
INDEX_FILE = "/tmp/opencode/shakespeare/index.pkl"

def b64_image(path):
    """Load image, convert to PNG (sglang OCR has a JPEG bug), return base64 data URL."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ocr_page(path):
    """OCR a single page image via Shadow API."""
    url = f"{BASE}/chat/completions"
    payload = {
        "model": "deepseek-ocr-2",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64_image(path)}},
            {"type": "text", "text": "Transcribe all text in this image. Output only the transcribed text."}
        ]}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            text = j["choices"][0]["message"].get("content", "") or ""
            return text.strip()
        except Exception as e:
            if attempt == 2:
                return f"[OCR ERROR: {e}]"
            time.sleep(1)

def embed_batch(texts):
    """Embed a batch of texts via moe directly."""
    url = f"{EMB_BASE}/embeddings"
    payload = {"model": "qwen3-embedding-0.6b", "input": texts}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
            r.raise_for_status()
            j = r.json()
            return [d["embedding"] for d in j["data"]]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)

def rerank(query, doc):
    """Rerank a query-doc pair via Shadow API. Returns P(yes)."""
    SYS = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    user = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc[:2000]}"
    url = f"{RERANK_BASE}/chat/completions"
    payload = {"model": "qwen3-reranker-0.6b", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user}
    ], "max_tokens": 8, "temperature": 0.0, "logprobs": True, "top_logprobs": 10}
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        lps = (j["choices"][0].get("logprobs") or {}).get("content") or []
        if lps:
            for t in (lps[0].get("top_logprobs") or []):
                if t.get("token", "").lower().lstrip().startswith("yes"):
                    return math.exp(t.get("logprob", -999))
        return 0.0
    except:
        return 0.0

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def main():
    # Get all page files
    pages = sorted(glob.glob(f"{PAGES_DIR}/page_*.jpg"))
    total = len(pages)
    print(f"=" * 70)
    print(f"  SHAKESPEARE OCR PIPELINE — {total} pages")
    print(f"  OCR → Embed → Index → Query")
    print(f"  API: {BASE}")
    print(f"=" * 70)

    # ── Stage 1: OCR all pages ──
    print(f"\n── Stage 1: OCR ({total} pages via DeepSeek-OCR-2) ──")
    ocr_start = time.time()
    page_texts = {}
    empty_count = 0
    error_count = 0

    for i, path in enumerate(pages):
        pageno = int(os.path.basename(path).split("_")[1].split(".")[0])
        t0 = time.time()
        text = ocr_page(path)
        dt = time.time() - t0
        page_texts[pageno] = text

        if not text or text.startswith("[OCR ERROR"):
            error_count += 1
        elif len(text.strip()) < 10:
            empty_count += 1

        # Progress every page
        elapsed = time.time() - ocr_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0
        sample = text[:60].replace("\n", " ") if text else "(empty)"
        status = "ERR" if text.startswith("[OCR ERROR") else ("EMPTY" if len(text.strip()) < 10 else "OK")
        if (i + 1) % 10 == 0 or i < 5 or status == "ERR":
            print(f"  [{i+1:>4d}/{total}] {dt:.2f}s | {rate:.1f} pg/s | ETA {eta:.0f}s | {status} | {sample!r}", flush=True)

    ocr_total = time.time() - ocr_start
    total_chars = sum(len(t) for t in page_texts.values())
    print(f"\n  OCR complete: {ocr_total:.1f}s | {total/ocr_total:.1f} pages/s")
    print(f"  Total text: {total_chars:,} chars | avg {total_chars//total:,} chars/page")
    print(f"  Errors: {error_count} | Near-empty: {empty_count}")

    # ── Stage 2: Embed all pages ──
    print(f"\n── Stage 2: Embed ({total} pages via Qwen3-Embedding-0.6B) ──")
    emb_start = time.time()

    # Batch embed in groups of 32
    page_nums = sorted(page_texts.keys())
    all_vecs = {}
    batch_size = 32
    for i in range(0, len(page_nums), batch_size):
        batch_nums = page_nums[i:i+batch_size]
        batch_texts = [page_nums_to_text(page_texts, n) for n in batch_nums]
        vecs = embed_batch(batch_texts)
        for n, v in zip(batch_nums, vecs):
            all_vecs[n] = v
        if (i + batch_size) % 200 == 0 or i == 0:
            elapsed = time.time() - emb_start
            print(f"  embedded {min(i+batch_size, len(page_nums))}/{total} ({elapsed:.1f}s)")

    emb_total = time.time() - emb_start
    print(f"  Embed complete: {emb_total:.2f}s | {total/emb_total:.0f} pages/s")

    # ── Save index ──
    print(f"\n── Saving index to {INDEX_FILE} ──")
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"page_texts": page_texts, "page_vecs": all_vecs}, f)
    idx_size = os.path.getsize(INDEX_FILE) / 1024 / 1024
    print(f"  Saved: {idx_size:.1f} MB")

    # ── Stage 3: Query test ──
    print(f"\n── Stage 3: Query test (retrieve + rerank) ──")
    queries = [
        "To be or not to be, that is the question",
        "Friends, Romans, countrymen, lend me your ears",
        "O Romeo, Romeo, wherefore art thou Romeo",
        "A horse, a horse, my kingdom for a horse",
        "The course of true love never did run smooth",
        "All the world's a stage, and all the men and women merely players",
        "Is this a dagger which I see before me",
        "Now is the winter of our discontent made glorious summer by this sun of York",
    ]

    # Embed queries
    query_vecs = embed_batch(queries)

    for qi, query in enumerate(queries):
        # Retrieve top-5 by cosine
        scores = [(cosine(query_vecs[qi], all_vecs[n]), n) for n in page_nums]
        scores.sort(reverse=True)
        top5 = scores[:5]

        # Rerank top-5
        best_pyes = 0
        best_page = top5[0][1]
        for score, pageno in top5:
            pyes = rerank(query, page_texts[pageno])
            if pyes > best_pyes:
                best_pyes = pyes
                best_page = pageno

        # Show result
        snippet = page_texts[best_page][:120].replace("\n", " ")
        print(f"\n  Q: \"{query}\"")
        print(f"  → page {best_page} (cosine={scores[0][0]:.3f}, P(yes)={best_pyes:.3f})")
        print(f"    {snippet!r}...")

    # ── Summary ──
    total_time = ocr_total + emb_total
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Pages:              {total}")
    print(f"  OCR time:           {ocr_total:.1f}s ({total/ocr_total:.1f} pages/s)")
    print(f"  Embed time:         {emb_total:.1f}s ({total/emb_total:.0f} pages/s)")
    print(f"  Total:              {total_time:.1f}s ({total/total_time:.1f} pages/s)")
    print(f"  Text extracted:     {total_chars:,} chars ({total_chars/5:.0f} words approx)")
    print(f"  Index size:         {idx_size:.1f} MB")
    print(f"  OCR errors:         {error_count}")
    print(f"  Near-empty pages:   {empty_count} (likely blank/illustration pages)")

def page_nums_to_text(page_texts, n):
    t = page_texts.get(n, "")
    return t if t else "(blank page)"

if __name__ == "__main__":
    main()
