#!/usr/bin/env python3
"""Test 2: Qwen3-Embedding-0.6B semantic-meaningfulness validation.
Runs locally (numpy available), hits http://<MOE_HOST>:30001."""
import json, math, time, requests
import numpy as np

BASE="http://<MOE_HOST>:30001/v1/embeddings"
MODEL="qwen3-embedding-0.6b"

# Two semantically similar, one unrelated (per task spec)
S1="The cat sat on the mat."            # cat on mat
S2="A feline rested on the rug."        # paraphrase (feline=cat, rested=sat, rug=mat)
S3="Stock markets fell sharply today."  # unrelated (finance)
SENTS=[S1,S2,S3]

def embed(text):
    r=requests.post(BASE,json={"model":MODEL,"input":text},timeout=60)
    r.raise_for_status()
    j=r.json()
    d=j["data"][0]["embedding"]
    return d, j.get("model"), j.get("usage")

def cos(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)))

if __name__=="__main__":
    print("=== Test 2: Qwen3-Embedding-0.6B ===")
    vecs=[]; dim=None
    for s in SENTS:
        v,m,u=embed(s)
        vecs.append(v)
        if dim is None: dim=len(v)
        print(f"\nInput: {s!r}")
        print(f"  model={m} dim={len(v)} usage={u}")
        print(f"  vec[:3]={v[:3]}  ||v||={np.linalg.norm(v):.4f}")
    print(f"\nEmbedding dimension: {dim} (expected 1024 per memory file)")
    assert dim==1024, f"UNEXPECTED DIM {dim}"

    sim12=cos(vecs[0],vecs[1])  # similar pair (cat/feline)
    sim13=cos(vecs[0],vecs[2])  # vs unrelated (cat/stocks)
    sim23=cos(vecs[1],vecs[2])  # vs unrelated (feline/stocks)
    print("\n=== Cosine similarities ===")
    print(f"  sim(S1,S2) similar pair (cat / feline):       {sim12:.4f}")
    print(f"  sim(S1,S3) vs unrelated (cat / stocks):        {sim13:.4f}")
    print(f"  sim(S2,S3) vs unrelated (feline / stocks):     {sim23:.4f}")

    margin_a = sim12 - sim13
    margin_b = sim12 - sim23
    print(f"\n  margin(similar) - (S1 vs unrelated) = {margin_a:+.4f}")
    print(f"  margin(similar) - (S2 vs unrelated) = {margin_b:+.4f}")

    # Pass criterion: similar pair meaningfully higher than both unrelated pairs
    pass_a = sim12 > sim13 + 0.05
    pass_b = sim12 > sim23 + 0.05
    print(f"\n  similar > S1-vs-unrelated+0.05 : {pass_a}")
    print(f"  similar > S2-vs-unrelated+0.05 : {pass_b}")
    overall = pass_a and pass_b and sim12 > max(sim13,sim23)
    print(f"\n  >>> OVERALL: {'PASS' if overall else 'FAIL'} (similar pair must beat both unrelated)")

    with open("/tmp/opencode/embedding_results.json","w") as f:
        json.dump({"dim":dim,"sentences":SENTS,
                   "sim_12":sim12,"sim_13":sim13,"sim_23":sim23,
                   "margin_a":margin_a,"margin_b":margin_b,"pass":overall},f,indent=2)
