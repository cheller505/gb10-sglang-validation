#!/usr/bin/env python3
"""Verify the custom no-think chat-template fix for Qwen3-Reranker via /v1/chat/completions.
Tests against the temp instance on port 30003 (with --chat-template fix).
No special client kwargs — plain chat API, exactly as a normal client would call."""
import json, math, time, requests

BASE="http://192.168.200.135:30003/v1/chat/completions"
MODEL="qwen3-reranker-test"
INSTRUCT="Given a web search query, retrieve relevant passages that answer the query"
QUERY="How do I bake sourdough bread at home?"
DOC_REL=("Sourdough baking begins with a wild-yeast starter made from flour and water. "
  "After feeding the starter for several days, you mix it with more flour, water, and salt, "
  "then allow a long bulk fermentation before shaping, proofing, and baking in a hot Dutch oven "
  "to develop a crisp crust and open crumb.")
DOC_IRREL=("The 2024 Summer Olympics were held in Paris, France, featuring 32 sports and "
  "329 events across 48 disciplines. The United States topped the medal table with 126 total "
  "medals, followed by China and Japan.")
SYS=('Judge whether the Document meets the requirements based on the Query and the Instruct '
     'provided. Note that the answer can only be "yes" or "no".')
def fmt_user(q,d): return f"<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}"

def run(query,doc,label,logprobs=True):
    payload={"model":MODEL,
        "messages":[{"role":"system","content":SYS},
                    {"role":"user","content":fmt_user(query,doc)}],
        "max_tokens":8,"temperature":0.0,
        "logprobs":logprobs,"top_logprobs":10}
    t0=time.time(); r=requests.post(BASE,json=payload,timeout=120); dt=time.time()-t0
    print(f"\n=== {label} (HTTP {r.status_code}, {dt:.1f}s) ===")
    j=r.json(); ch=j["choices"][0]; out=ch["message"].get("content","")
    print("OUTPUT:",repr(out))
    print("finish:",ch.get("finish_reason"),"| usage:",j.get("usage",{}))
    yes_lp=None; no_lp=None
    if ch.get("logprobs"):
        lps=ch["logprobs"].get("content",[])
        if lps:
            print(f"pos0 sampled: {lps[0].get('token')!r} lp={lps[0].get('logprob'):.4f}")
            tops=lps[0].get("top_logprobs") or []
            print("  top_logprobs pos0:")
            for t in tops:
                tok=t.get('token',''); lp=t.get('logprob')
                tl=tok.lower().lstrip()
                if tl.startswith("yes") and yes_lp is None: yes_lp=lp
                if tl=="no" and no_lp is None: no_lp=lp
                print(f"    {tok!r:16s} lp={lp:.4f} prob={math.exp(lp):.4f}")
    return out, yes_lp, no_lp

print("### Verifying custom no-think chat-template fix (port 30003) ###")
print("### Plain /v1/chat/completions, NO special client kwargs ###")
out_rel,yes_rel,no_rel=run(QUERY,DOC_REL,"(a) RELEVANT — expect yes")
out_irr,yes_irr,no_irr=run(QUERY,DOC_IRREL,"(b) IRRELEVANT — expect no")

print("\n=== VERDICT ===")
pa=out_rel.strip().lower().startswith("yes")
pb=out_irr.strip().lower().startswith("no")
print(f"  (a) relevant->'yes': {pa} ({out_rel!r})")
print(f"  (b) irrelevant->'no': {pb} ({out_irr!r})")
print(f"  >>> FIX {'VERIFIED' if pa and pb else 'STILL FAILING'} <<<")
print(f"\n  'yes' logprob: relevant={yes_rel}  irrelevant={yes_irr}")
print(f"  'no'  logprob: relevant={no_rel}  irrelevant={no_irr}")
if yes_rel is not None and yes_irr is not None:
    print(f"  P(yes) relevant={math.exp(yes_rel):.4f}  irrelevant={math.exp(yes_irr):.4f}")
    print(f"  >>> 'yes' prob higher for relevant: {math.exp(yes_rel)>math.exp(yes_irr)}")

with open("/tmp/opencode/reranker_fix_verify.json","w") as f:
    json.dump({"out_rel":out_rel,"out_irr":out_irr,"pass_a":pa,"pass_b":pb,
               "yes_lp_rel":yes_rel,"yes_lp_irr":yes_irr,
               "no_lp_rel":no_rel,"no_lp_irr":no_irr},f,indent=2)
