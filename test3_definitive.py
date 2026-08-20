#!/usr/bin/env python3
"""Definitive Test 3: correct think tokens (built from codepoints) + chat_template_kwargs probe."""
import json, math, time, requests

COMP="http://192.168.200.135:30002/v1/completions"
CHAT="http://192.168.200.135:30002/v1/chat/completions"
MODEL="qwen3-reranker-0.6b"
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

# Build think tokens from codepoints to avoid any encoding confusion
# open = <think> (3c 74 68 69 6e 6b 3e), close = </think> (3c 2f 74 68 69 6e 6b 3e)
THINK_OPEN="".join(chr(c) for c in [0x3c,0x74,0x68,0x69,0x6e,0x6b,0x3e])
THINK_CLOSE="".join(chr(c) for c in [0x3c,0x2f,0x74,0x68,0x69,0x6e,0x6b,0x3e])
print(f"Think tokens: open={THINK_OPEN!r} close={THINK_CLOSE!r}")

def official_prompt(q,d):
    # Exact Qwen3-Reranker format: assistant prefix + empty think block
    return (f'<|im_start|>system\n{SYS}<|im_end|>\n'
            f'<|im_start|>user\n{fmt_user(q,d)}<|im_end|>\n'
            f'<|im_start|>assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n')

def raw(q,d,label,max_tokens=20):
    payload={"model":MODEL,"prompt":official_prompt(q,d),"max_tokens":max_tokens,
             "temperature":0.0,"logprobs":True,"top_logprobs":10}
    t0=time.time(); r=requests.post(COMP,json=payload,timeout=180); dt=time.time()-t0
    print(f"\n=== RAW {label} (HTTP {r.status_code}, {dt:.1f}s) ===")
    j=r.json(); ch=j["choices"][0]; out=ch.get("text","")
    print("finish:",ch.get("finish_reason"),"| usage:",j.get("usage",{}))
    print("OUTPUT:",repr(out))
    lps=ch.get("logprobs",{}).get("content") or []
    yes_lp=None; no_lp=None
    if lps:
        print(f"pos0 sampled: {lps[0].get('token')!r} lp={lps[0].get('logprob'):.4f}")
        tops=lps[0].get("top_logprobs") or []
        print("  top_logprobs pos0:")
        for t in tops:
            tok=t.get('token',''); lp=t.get('logprob')
            tl=tok.lower().lstrip()
            if tl.startswith("yes") and yes_lp is None: yes_lp=lp
            if tl=="no" and no_lp is None: no_lp=lp
            print(f"    {tok!r:18s} lp={lp:.4f} prob={math.exp(lp):.4f}")
    return out, yes_lp, no_lp

print("\n##### A. Definitive raw /v1/completions with CORRECT think tokens #####")
out_rel,yes_rel,no_rel=raw(QUERY,DOC_REL,"RELEVANT — expect yes")
out_irr,yes_irr,no_irr=raw(QUERY,DOC_IRREL,"IRRELEVANT — expect no")
pa=out_rel.strip().lower().startswith("yes"); pb=out_irr.strip().lower().startswith("no")
print(f"\n  (a) relevant->'yes': {pa} ({out_rel!r})  (b) irrelevant->'no': {pb} ({out_irr!r})")
print(f"  MODEL(raw,correct format): {'PASS' if pa and pb else 'FAIL'}")
print(f"  'yes' lp: relevant={yes_rel} irrelevant={yes_irr}")
print(f"  'no'  lp: relevant={no_rel} irrelevant={no_irr}")
if yes_rel is not None:
    pr=math.exp(yes_rel); pi=math.exp(yes_irr) if yes_irr is not None else 0
    print(f"  P(yes): relevant={pr:.4f} irrelevant={pi:.4f} -> higher for relevant: {pr>pi}")

print("\n##### B. Probe: does /v1/chat/completions honor chat_template_kwargs.enable_thinking? #####")
def chat_probe(label,doc,extra):
    payload={"model":MODEL,"messages":[{"role":"system","content":SYS},
        {"role":"user","content":fmt_user(QUERY,doc)}],"max_tokens":20,"temperature":0.0,
        "logprobs":True,"top_logprobs":10, **extra}
    r=requests.post(CHAT,json=payload,timeout=120); j=r.json()
    if "choices" not in j:
        print(f"  {label}: ERROR {json.dumps(j)[:300]}"); return None
    ch=j["choices"][0]; out=ch["message"].get("content","")
    lps=(ch.get("logprobs") or {}).get("content") or []
    tl0=(lps[0].get("token","") if lps else "")
    print(f"  {label}: out={out!r} finish={ch.get('finish_reason')} pos0={tl0!r}")
    return out
for extra,label in [
    ({"chat_template_kwargs":{"enable_thinking":False}},"chat_template_kwargs.enable_thinking=False"),
    ({"enable_thinking":False},"top-level enable_thinking=False"),
    ({"chat_template_kwargs":{"enable_thinking":True}},"enable_thinking=True (default, control)"),
]:
    chat_probe(f"RELEVANT ({label})",DOC_REL,extra)
    chat_probe(f"IRRELEVANT ({label})",DOC_IRREL,extra)
