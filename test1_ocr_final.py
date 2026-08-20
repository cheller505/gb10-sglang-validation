#!/usr/bin/env python3
"""Clean canonical Test 1 + text-only check + dimension-sensitivity probe."""
import base64, json, os, time, urllib.request, urllib.error
from PIL import Image, ImageDraw, ImageFont

BASE="http://localhost:30000/v1/chat/completions"; MODEL="deepseek-ocr-2"
SANS="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SERIF="/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
OUT="/tmp/ocr-final"; os.makedirs(OUT, exist_ok=True)

def b64(p): return "data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode()
def post(payload, timeout=180):
    req=urllib.request.Request(BASE,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r: return r.status,r.read().decode(),time.time()-t0
    except urllib.error.HTTPError as e: return e.code,e.read().decode(),time.time()-t0
    except Exception as e: return -1,str(e),time.time()-t0

def ocr(path,label,max_tokens=512,prompt="Transcribe all text visible in this image exactly as it appears. Output only the transcribed text, nothing else."):
    payload={"model":MODEL,"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":b64(path)}},
        {"type":"text","text":prompt}]}],"max_tokens":max_tokens,"temperature":0.0}
    sc,body,dt=post(payload)
    try: j=json.loads(body)
    except: print(f"\n=== {label} === NON-JSON: {body[:400]}"); return None
    ch=j.get("choices",[{}])[0]; out=ch.get("message",{}).get("content")
    print(f"\n=== {label} === (HTTP {sc}, {dt:.1f}s)")
    print("finish_reason:", ch.get("finish_reason"), "| matched_stop:", ch.get("matched_stop"), "| usage:", j.get("usage",{}))
    print("OUTPUT:", repr(out))
    return out

# ---- Image 1 (simple synthetic): pangram, properly sized ----
IMG1="The quick brown fox jumps over the lazy dog."
img=Image.new("RGB",(1200,160),"white"); d=ImageDraw.Draw(img); f=ImageFont.truetype(SANS_B,36)
bb=d.textbbox((0,0),IMG1,font=f); w=bb[2]-bb[0]; h=bb[3]-bb[1]
d.text(((1200-w)/2-bb[0],(160-h)/2-bb[1]),IMG1,fill="black",font=f)
img.save(OUT+"/img1_simple.png")

# ---- Image 2 (realistic document-style) ----
IMG2_GT=["QUARTERLY REPORT","Fiscal Year 2026 - Q2 Summary",
 "Revenue increased by 17 percent year over year, driven by strong demand in the embedded systems market. Operating margins improved to 23.4 percent.",
 "Confidential - Internal Use Only"]
W,H=960,540; img=Image.new("RGB",(W,H),"#fafafa"); d=ImageDraw.Draw(img)
d.rectangle([0,0,W,78],fill="#1f3a5f"); ft=ImageFont.truetype(SANS_B,42); t="QUARTERLY REPORT"
bb=d.textbbox((0,0),t,font=ft); d.text(((W-(bb[2]-bb[0]))/2-bb[0],16-bb[1]),t,fill="white",font=ft)
cy=96; d.text((70,cy),"Fiscal Year 2026 - Q2 Summary",fill="#222222",font=ImageFont.truetype(SANS_B,26)); cy+=48
d.line([(70,cy),(W-70,cy)],fill="#cccccc",width=2); cy+=24
for line in ["Revenue increased by 17 percent year over year,","driven by strong demand in the embedded systems","market. Operating margins improved to 23.4 percent."]:
    d.text((70,cy),line,fill="#111111",font=ImageFont.truetype(SERIF,26)); bb=d.textbbox((0,0),line,font=ImageFont.truetype(SERIF,26)); cy+=(bb[3]-bb[1])+12
cy+=28; d.text((70,cy),"Confidential - Internal Use Only",fill="#888888",font=ImageFont.truetype(SANS,18))
img.save(OUT+"/img2_document.png")

print("GT1:",repr(IMG1))
print("GT2:",IMG2_GT)
o1=ocr(OUT+"/img1_simple.png","IMG1 simple synthetic (pangram)")
o2=ocr(OUT+"/img2_document.png","IMG2 realistic document-style")

# ---- text-only degradation ----
payload={"model":MODEL,"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0.0}
sc,body,dt=post(payload,timeout=120)
print(f"\n=== TEXT-ONLY (no image) degradation check === (HTTP {sc}, {dt:.1f}s)")
try:
    j=json.loads(body); out=j["choices"][0]["message"].get("content")
    print("finish_reason:",j["choices"][0].get("finish_reason"),"| usage:",j.get("usage",{}))
    print("OUTPUT (nonsensical OK, error=BAD):",repr(out))
    to_out=out; to_sc=sc
except Exception as e:
    print("ERROR/non-JSON:",sc,body[:400]); to_out=None; to_sc=sc

# ---- dimension-sensitivity probe (to characterize the empty-response flag) ----
print("\n--- dimension sensitivity probe (pangram, varied dims) ---")
for (Wd,Hd,sz) in [(800,200,48),(1100,200,52),(1200,160,36),(900,300,40),(1400,300,48)]:
    p=OUT+f"/probe_{Wd}x{Hd}_s{sz}.png"
    img=Image.new("RGB",(Wd,Hd),"white"); d=ImageDraw.Draw(img); f=ImageFont.truetype(SANS_B,sz)
    bb=d.textbbox((0,0),IMG1,font=f); w=bb[2]-bb[0]; h=bb[3]-bb[1]
    d.text(((Wd-w)/2-bb[0],(Hd-h)/2-bb[1]),IMG1,fill="black",font=f); img.save(p)
    ocr(p,f"pangram {Wd}x{Hd} size{sz}",max_tokens=256,prompt="Transcribe the text in this image.")

with open(OUT+"/results.json","w") as f:
    json.dump({"img1_gt":IMG1,"img1_out":o1,"img2_gt":IMG2_GT,"img2_out":o2,
               "textonly_status":to_sc,"textonly_out":to_out},f,indent=2)
print("\nSaved "+OUT+"/results.json and images in "+OUT)
