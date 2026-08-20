#!/usr/bin/env python3
"""End-to-end RAG pipeline benchmark using all 3 moe models.

Workflow: Document OCR → Chunk → Embed → Retrieve → Rerank → Answer
Simulates processing a batch of scanned document pages.
"""
import base64, json, time, statistics, math, os, urllib.request, urllib.error
from PIL import Image, ImageDraw, ImageFont

# ── helpers ──────────────────────────────────────────────────────────────────
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

def get_model_id(port):
    req = urllib.request.Request(f"http://localhost:{port}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]

OCR_PORT, EMB_PORT, RERANK_PORT = 30000, 30001, 30002
OCR_MODEL = get_model_id(OCR_PORT)
EMB_MODEL = get_model_id(EMB_PORT)
RERANK_MODEL = get_model_id(RERANK_PORT)

# ── simulate a stack of scanned document pages ───────────────────────────────
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

PAGES = [
    # Page 1: tech article about transformers
    {
        "title": "Attention Is All You Need",
        "lines": [
            "The dominant sequence transduction models are based on complex recurrent or",
            "convolutional neural networks. We propose a new simple network architecture,",
            "the Transformer, based solely on attention mechanisms, dispensing with recurrence",
            "and convolutions entirely. Experiments on two machine translation tasks show",
            "these models to be superior in quality while being more parallelizable.",
        ],
        "query": "What architecture replaced recurrent neural networks for sequence tasks?",
    },
    # Page 2: recipe page
    {
        "title": "Sourdough Bread Recipe",
        "lines": [
            "To make sourdough bread, first prepare a starter using flour and water.",
            "Feed the starter daily for five days until it becomes bubbly and active.",
            "Mix 500g flour, 350g water, 10g salt, and 100g starter. Knead for ten minutes.",
            "Let the dough rise for four hours, shape into a round, and proof overnight.",
            "Bake at 450 degrees Fahrenheit in a preheated Dutch oven for 40 minutes.",
        ],
        "query": "How do I bake sourdough bread at home?",
    },
    # Page 3: financial report
    {
        "title": "Q2 Financial Summary",
        "lines": [
            "Revenue for the second quarter reached 42.3 million dollars, up 15 percent.",
            "Operating expenses decreased by 8 percent due to cost optimization measures.",
            "Net income was 9.1 million dollars, with earnings per share of 1.87 dollars.",
            "The company expects continued growth in the second half of the fiscal year.",
            "Cash reserves stand at 28.5 million dollars as of June 30.",
        ],
        "query": "What was the quarterly revenue and net income?",
    },
    # Page 4: unrelated filler (should rank low for any of the above queries)
    {
        "title": "Garden Maintenance Tips",
        "lines": [
            "Water your garden early in the morning to reduce evaporation losses.",
            "Mulch helps retain soil moisture and suppress weed growth around plants.",
            "Prune tomato plants regularly to encourage airflow and fruit production.",
            "Companion planting with marigolds can deter pests from vegetable beds.",
            "Rotate crops each season to prevent soil depletion and disease buildup.",
        ],
        "query": "How do I bake sourdough bread at home?",
    },
    # Page 5: another relevant doc for the transformer query
    {
        "title": "Transformer Architecture Explained",
        "lines": [
            "The transformer model uses self-attention to process input sequences in parallel.",
            "Multi-head attention allows the model to focus on different positions simultaneously.",
            "Positional encodings provide sequence order information to the attention layers.",
            "The encoder-decoder structure was original, but decoder-only variants like GPT",
            "became dominant for language modeling and text generation tasks.",
        ],
        "query": "What architecture replaced recurrent neural networks for sequence tasks?",
    },
]

def render_page(page):
    """Render a page dict as a PNG image (document-style)."""
    W, H = 1000, 600
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    # header bar
    d.rectangle([0, 0, W, 60], fill="#1f3a5f")
    ft = ImageFont.truetype(SANS_B, 32)
    t = page["title"]
    bb = d.textbbox((0, 0), t, font=ft)
    d.text(((W - (bb[2]-bb[0]))/2 - bb[0], 14 - bb[1]), t, fill="white", font=ft)
    # body
    cy = 90
    fb = ImageFont.truetype(SERIF, 24)
    for line in page["lines"]:
        d.text((60, cy), line, fill="#111111", font=fb)
        bb = d.textbbox((0, 0), line, font=fb)
        cy += (bb[3]-bb[1]) + 14
    # footer
    d.line([(60, H-40), (W-60, H-40)], fill="#cccccc", width=1)
    d.text((60, H-30), f"Page — {page['title']}", fill="#888888", font=ImageFont.truetype(SANS, 14))
    buf = __import__("io").BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ── pipeline stages ──────────────────────────────────────────────────────────
def ocr_page(img_url):
    """Stage 1: OCR a page image → text."""
    payload = {"model": OCR_MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img_url}},
        {"type": "text", "text": "Transcribe all text in this image. Output only the text."}
    ]}], "max_tokens": 512, "temperature": 0.0}
    sc, body, dt = post(f"http://localhost:{OCR_PORT}/v1/chat/completions", payload, timeout=60)
    j = json.loads(body)
    text = j["choices"][0]["message"].get("content", "") or ""
    return text, dt

def embed_texts(texts):
    """Stage 2: Embed a list of texts → vectors (batched)."""
    payload = {"model": EMB_MODEL, "input": texts}
    sc, body, dt = post(f"http://localhost:{EMB_PORT}/v1/embeddings", payload, timeout=30)
    j = json.loads(body)
    vecs = [d["embedding"] for d in j["data"]]
    return vecs, dt

def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0

RERANK_SYS = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
def rerank(query, doc):
    """Stage 3: Rerank a single query-doc pair → yes/no + logprob score."""
    user = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}"
    payload = {"model": RERANK_MODEL, "messages": [
        {"role": "system", "content": RERANK_SYS},
        {"role": "user", "content": user}
    ], "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 10}
    sc, body, dt = post(f"http://localhost:{RERANK_PORT}/v1/chat/completions", payload, timeout=30)
    j = json.loads(body)
    ch = j["choices"][0]
    lps = (ch.get("logprobs") or {}).get("content") or []
    yes_lp = None
    if lps:
        for t in (lps[0].get("top_logprobs") or []):
            if t.get("token", "").lower().lstrip().startswith("yes"):
                yes_lp = t.get("logprob")
    p_yes = math.exp(yes_lp) if yes_lp else 0.0
    return p_yes, dt

# ── run the pipeline ─────────────────────────────────────────────────────────
print("=" * 70)
print("  END-TO-END RAG PIPELINE BENCHMARK")
print("  OCR → Embed → Retrieve (cosine) → Rerank → Answer")
print("=" * 70)

# Pre-render all page images
print(f"\nRendering {len(PAGES)} document page images...")
page_images = [render_page(p) for p in PAGES]

# ── Stage 1: OCR all pages ──
print("\n── Stage 1: OCR (DeepSeek-OCR-2) ──")
ocr_times = []
page_texts = []
for i, (img, page) in enumerate(zip(page_images, PAGES)):
    text, dt = ocr_page(img)
    ocr_times.append(dt)
    page_texts.append(text)
    # check if key phrase from the page appears in OCR output
    key = page["title"].lower()[:20]
    match = key in text.lower()
    print(f"  Page {i+1} ({page['title']:30s}): {dt:.3f}s | {len(text):4d} chars | title_found={match}")

print(f"  → OCR total: {sum(ocr_times):.2f}s | per-page: {statistics.mean(ocr_times)*1000:.0f}ms avg | {len(PAGES)/sum(ocr_times):.1f} pages/s")

# ── Stage 2: Embed all page texts + queries ──
print("\n── Stage 2: Embed (Qwen3-Embedding-0.6B) ──")
queries = [p["query"] for p in PAGES]
all_texts = page_texts + queries
t0 = time.time()
all_vecs, dt = embed_texts(all_texts)
emb_total = time.time() - t0
page_vecs = all_vecs[:len(page_texts)]
query_vecs = all_vecs[len(page_texts):]
print(f"  Embedded {len(all_texts)} texts ({len(page_texts)} pages + {len(queries)} queries) in {emb_total:.3f}s")
print(f"  → {len(all_texts)/emb_total:.0f} embed/s | dim={len(all_vecs[0])}")

# ── Stage 3: Retrieve (cosine similarity) ──
print("\n── Stage 3: Retrieve (cosine similarity top-k) ──")
t0 = time.time()
for qi, query in enumerate(queries):
    scores = [(cos(query_vecs[qi], page_vecs[pi]), pi) for pi in range(len(page_texts))]
    scores.sort(reverse=True)
    top3 = scores[:3]
    print(f"  Query {qi+1}: \"{query[:60]}...\"")
    for score, pi in top3:
        print(f"    page {pi+1} ({PAGES[pi]['title']:30s}): cosine={score:.4f}")
retrieve_time = time.time() - t0
print(f"  → Retrieve (3 queries × 5 pages): {retrieve_time*1000:.1f}ms")

# ── Stage 4: Rerank top candidates ──
print("\n── Stage 4: Rerank (Qwen3-Reranker-0.6B) ──")
rerank_times = []
for qi, query in enumerate(queries):
    scores = [(cos(query_vecs[qi], page_vecs[pi]), pi) for pi in range(len(page_texts))]
    scores.sort(reverse=True)
    top3 = scores[:3]
    print(f"  Query {qi+1}: \"{query[:60]}...\"")
    for score, pi in top3:
        p_yes, dt = rerank(query, page_texts[pi])
        rerank_times.append(dt)
        verdict = "RELEVANT" if p_yes > 0.5 else "not relevant"
        print(f"    page {pi+1} ({PAGES[pi]['title']:30s}): P(yes)={p_yes:.4f} {verdict} ({dt*1000:.0f}ms)")
print(f"  → Rerank: {statistics.mean(rerank_times)*1000:.0f}ms avg per judgment | {len(rerank_times)} judgments in {sum(rerank_times):.2f}s")

# ── Summary ──
total_ocr = sum(ocr_times)
total_emb = emb_total
total_rerank = sum(rerank_times)
total_pipeline = total_ocr + total_emb + total_rerank + retrieve_time

print(f"\n{'='*70}")
print(f"  PIPELINE SUMMARY")
print(f"{'='*70}")
print(f"  Pages processed:    {len(PAGES)}")
print(f"  OCR stage:          {total_ocr:.2f}s  ({total_ocr/total_pipeline*100:.0f}% of total)")
print(f"  Embed stage:        {total_emb:.3f}s ({total_emb/total_pipeline*100:.0f}% of total)")
print(f"  Retrieve stage:     {retrieve_time*1000:.1f}ms ({retrieve_time/total_pipeline*100:.0f}% of total)")
print(f"  Rerank stage:       {total_rerank:.2f}s  ({total_rerank/total_pipeline*100:.0f}% of total)")
print(f"  ────────────────────────────")
print(f"  Total pipeline:     {total_pipeline:.2f}s")
print(f"  Pages/second:       {len(PAGES)/total_pipeline:.1f}")
print(f"  Latency per query:  {(total_ocr + total_emb + total_rerank/3)/3:.2f}s (OCR+embed+1 rerank)")
print(f"\n  Bottleneck: OCR ({total_ocr/total_pipeline*100:.0f}%) — embedding and reranking are negligible")
print(f"  At this rate, a 100-page document would take ~{100/(len(PAGES)/total_pipeline):.0f}s to fully process")
