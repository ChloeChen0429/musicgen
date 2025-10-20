#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Emotion → Chord Performer (End-Gated Detection, No Pandas)

- End-gated emotion sampling (near block end), vote once → NEXT chord.
- At least 1 drum + 1 tonal layer each block.
- Uses loops under the “Berlin underground” root folder.
- *** Modified: build emotion→chord POOLS from your loop_meta_v2.csv / library (includes sharps), 7 emotions only.
"""

import os, sys, time, math, threading, queue, random, csv, re
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment, effects
from pydub.effects import low_pass_filter, high_pass_filter
from pydub.generators import Sine

# ========= Paths =========
ENV_AUDIO_DIR = os.environ.get("TECHNO_AUDIO_DIR","").strip()
CANDIDATE_AUDIO_ROOTS = [
    ENV_AUDIO_DIR,
    "/Volumes/TOSHIBAEXT/ualproject/webapp/data/Berlin underground",
    "/Volumes/TOSHIBAEXT/ualproject/dataset/techno dataset/Berlin underground",
]
CANDIDATE_DATA_DIRS = [
    "/Volumes/TOSHIBAEXT/ualproject/webapp/data",
    "/Volumes/TOSHIBAEXT/ualproject/dataset/techno dataset",
]
MODEL_PATH = "/Volumes/TOSHIBAEXT/ualproject/webapp/models/club_trainaug_best_20251005-203557.pt"
AUDIO_EXTS = {".wav",".wave",".aif",".aiff",".mp3",".flac",".m4a"}

def count_audio_files(root: str) -> int:
    if not root or not os.path.isdir(root): return 0
    c=0
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.startswith("._"): continue
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS: c+=1
    return c

def pick_audio_root() -> str:
    scored=[]
    for p in CANDIDATE_AUDIO_ROOTS:
        n=count_audio_files(p)
        if n>0: scored.append((n,p))
    if not scored:
        raise RuntimeError("❌ No audio found. Set env var TECHNO_AUDIO_DIR to your 'Berlin underground' folder.")
    scored.sort(reverse=True)
    n,p=scored[0]
    print(f"📂 Using audio root: {p} (files: {n})")
    return p

AUDIO_DIR = pick_audio_root()

def pick_data_dir() -> str:
    for d in CANDIDATE_DATA_DIRS:
        if os.path.isdir(d): return d
    return os.path.dirname(os.path.dirname(AUDIO_DIR))

DATA_DIR  = pick_data_dir()
LOOPS_CSV = os.path.join(DATA_DIR, "loop_meta_v2.csv")      # optional
AUTO_CSV  = os.path.join(DATA_DIR, "loop_meta_v2_autogen.csv")

# ========= Show parameters =========
BPM=130
BARS_PER_BLOCK=8
CROSSFADE_MS=1200
PREBUILD_SEC=2.0            # build next block this much earlier
DETECT_WIN_SEC=0.8          # emotion “gate” window near block end
SR=44100
CALM_CHORD="Am"
DEFAULT_ENERGY=0.60
TEMP=0.7
NO_REPEAT_MEMORY=6
VAR=0.2
SEED=None

OUTPUT_GAIN_DB=6
NORMALIZE_BLOCK=False

# >>> MOD START: remove old single mapping; we'll build pools dynamically
# EMO_TO_CHORD={"sad":"C","frown":"C","neutral":"Am","happy":"G","smile":"G","surprised":"Dm","angry":"F"}
# Base (semantic) chord templates per emotion; will be filtered/expanded by actual library roots.
EMO_TO_CHORDS_BASE = {
    "neutral":   ["Am","C","Em","Dm"],           
    "happy":     ["C","G","D","A"],              
    "surprised": ["G","A","D","E"],            
    "sad":       ["E","A","D","G"],              
    "angry":     ["E","B","F#","C#"],            
    "fear":      ["A","E","B","F#"],            
    "disgust":   ["D","A","E","B"],              
}
# <<< MOD END

# ========= Playback & helpers =========
try:
    from pydub.playback import _play_with_simpleaudio as play_sa
    HAVE_SA=True
except Exception:
    HAVE_SA=False
    from pydub.playback import play as play_ffplay
    def play_sa(seg):
        threading.Thread(target=lambda: play_ffplay(seg), daemon=True).start()
        class Dummy: pass
        return Dummy()

def audio_self_test():
    print("🔊 Audio self-test (440Hz, 0.5s)…")
    try:
        tone = Sine(440).to_audio_segment(duration=500).apply_gain(-8)
        play_sa(tone); time.sleep(0.55)
        print("✅ Audio OK. If you didn’t hear it, check output device/volume.")
    except Exception as e:
        print("⚠️ Self-test failed:", e)

def play_block(seg: AudioSegment):
    if OUTPUT_GAIN_DB: seg=seg.apply_gain(OUTPUT_GAIN_DB)
    if NORMALIZE_BLOCK: seg=effects.normalize(seg)
    return play_sa(seg)

KEYS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
FLATS={"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
TONAL={"Bass","Pad","Synth","Lead","Arp","Chord","Keys","Pluck","Melody"}  # >>> MOD: 扩展有音高类型
CORE ={"Kick","Clap","Drum"}
ORN  ={"Top","Ride","Perc","SFX","Other"}

def clamp(x,a,b): 
    return a if x<a else (b if x>b else x)

def key_to_num(k: Optional[str]) -> Optional[int]:
    if not isinstance(k,str) or not k: return None
    k = FLATS.get(k.strip(),k.strip()).replace('m','')
    return KEYS.index(k) if k in KEYS else None

def semitone_shift(src: Optional[str], tgt: Optional[str]) -> int:
    s,t=key_to_num(src), key_to_num(tgt)
    if s is None or t is None: return 0
    d=t-s
    if d>6: d-=12
    elif d<-6: d+=12
    return d

def load_audio(p: str) -> Optional[AudioSegment]:
    if os.path.basename(p).startswith("._"): return None
    try:
        seg=AudioSegment.from_file(p).set_frame_rate(SR).set_channels(2)
        return seg if len(seg)>=200 else None
    except Exception:
        return None

def pitch_shift(seg: AudioSegment, semis: int) -> AudioSegment:
    if seg is None or semis==0: return seg
    rate=2**(semis/12)
    return seg._spawn(seg.raw_data, overrides={'frame_rate': int(seg.frame_rate*rate)}).set_frame_rate(seg.frame_rate)

def random_slice(seg: AudioSegment, window_ms: int) -> AudioSegment:
    if seg is None: return seg
    if len(seg)<=window_ms+400: return seg[:window_ms]
    start=random.randint(0, max(0,len(seg)-window_ms))
    return seg[start:start+window_ms]

def tile_to(seg: AudioSegment, target_ms: int) -> AudioSegment:
    if seg is None: return AudioSegment.silent(duration=target_ms)
    seg=random_slice(seg, min(len(seg), target_ms))
    if len(seg)>=target_ms: return seg[:target_ms]
    reps=max(2, math.ceil(target_ms/max(1,len(seg))))
    out=seg
    for _ in range(reps-1):
        out=out.append(seg, crossfade=min(60, len(seg)//6 if len(seg)>0 else 0))
    return out[:target_ms]

def softmax_scores(scores: List[float], t: float) -> List[float]:
    if not scores: return []
    if t<=1e-6:
        best=min(scores)
        return [1.0 if s==best else 0.0 for s in scores]
    mx=max(scores)
    exps=[math.exp(-(s-mx)/float(t)) for s in scores]
    s=sum(exps) or 1.0
    return [e/s for e in exps]

class RecentMem:
    def __init__(self, cap=6): self.cap=cap; self.buf=[]
    def push(self, paths: List[str]):
        for p in paths: self.buf.append(p)
        if len(self.buf)>self.cap: self.buf=self.buf[-self.cap:]
    def penal(self, paths: List[str]) -> List[float]:
        s=set(self.buf); return [0.15 if p in s else 1.0 for p in paths]

# ========= Loop metadata (pure Python) =========
KEY_RE=re.compile(r'(?<![A-Za-z])([A-G](?:#|b)?m?)(?![A-Za-z])', re.IGNORECASE)
def parse_key_from_name(name:str)->str:
    m=KEY_RE.search(name.replace("_"," ").replace("-"," "))
    if not m: return "C"
    tok=m.group(1); tok=tok[0].upper()+tok[1:]
    tok=tok.replace("b","#")
    return FLATS.get(tok,tok)

def guess_type(path: str, given: str="") -> str:
    t=(given or "").lower()
    f=os.path.basename(path).lower()
    parent=os.path.basename(os.path.dirname(path)).lower()
    tokens=" ".join([t,f,parent])
    def has(*ks): return any(k in tokens for k in ks)
    if has("bass","sub"): return "Bass"
    if has("pad","chord","atm"): return "Pad"
    if has("synth","lead","arp"):return "Synth"
    if has("kick","bd"):         return "Kick"
    if has("clap","snare","rim"):return "Clap"
    if has("drumloop","drum","loop"): return "Drum"
    if has("hat","hh","ride","top"):  return "Top"
    if has("perc","tom","clave"):     return "Perc"
    if has("sfx","noise","impact"):   return "SFX"
    return (given or "Other").title()

def estimate_energy_quick(path:str)->float:
    seg=load_audio(path)
    if seg is None: return 0.5
    if len(seg)>8000: seg=seg[:8000]
    db = seg.dBFS if seg.dBFS!=float("-inf") else -60.0
    return clamp((db+60.0)/60.0, 0.0, 1.0)

def read_loops_csv(path:str) -> List[Dict]:
    rows=[]
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        for r in reader:
            p=str(r.get("path","")).strip()
            if not p: continue
            bn=os.path.basename(p)
            # remap to the actual root by filename
            fixed=""
            for root,_,fs in os.walk(AUDIO_DIR):
                for fn in fs:
                    if fn.startswith("._"): continue
                    if os.path.splitext(fn)[1].lower() not in AUDIO_EXTS: continue
                    if fn.lower()==bn.lower():
                        fixed=os.path.join(root,fn); break
                if fixed: break
            if not fixed or not os.path.exists(fixed): continue
            typ = guess_type(fixed, r.get("type",""))
            key = str(r.get("key_guess","C")).strip() or "C"
            key = key.replace("b","#")
            try:
                en=float(r.get("energy", "0.5"))
            except Exception:
                en=0.5
            rows.append({"path":fixed, "type":typ, "key_guess":key, "energy":en})
    return rows

def index_audio_library(root:str) -> List[Dict]:
    print("📦 Scanning audio library …")
    out=[]
    for r,_,fs in os.walk(root):
        for fn in fs:
            if fn.startswith("._"): continue
            if os.path.splitext(fn)[1].lower() not in AUDIO_EXTS: continue
            p=os.path.join(r,fn)
            typ=guess_type(p,"")
            key=parse_key_from_name(fn) if typ in TONAL else "C"
            en=estimate_energy_quick(p)
            out.append({"path":p, "type":typ, "key_guess":key, "energy":en})
    print(f"✅ Indexed {len(out)} loops")
    try:
        with open(AUTO_CSV, "w", encoding="utf-8", newline="") as f:
            w=csv.DictWriter(f, fieldnames=["path","type","key_guess","energy"])
            w.writeheader()
            for r in out: w.writerow(r)
        print("💾 Saved fallback CSV:", AUTO_CSV)
    except Exception as e:
        print("⚠️ Couldn’t save fallback CSV:", e)
    return out

def load_loops() -> List[Dict]:
    rows=[]
    if os.path.isfile(LOOPS_CSV):
        try:
            rows=read_loops_csv(LOOPS_CSV)
            if rows:
                print(f"✅ Loaded {len(rows)} loops from CSV (limited to {AUDIO_DIR})")
        except Exception as e:
            print("⚠️ CSV read failed, falling back to scan:", e)
    if not rows:
        rows=index_audio_library(AUDIO_DIR)
    rows=[r for r in rows if r["path"].startswith(AUDIO_DIR) and os.path.exists(r["path"])]
    if not rows: raise RuntimeError("No usable loops under the root.")
    return rows

# >>> MOD START: chord-pool builder (no pandas)
from collections import Counter, defaultdict

def _root_from_token(token:str)->Optional[str]:
    t=(token or "").upper().replace("MIN","").replace("MAJ","")
    t=t.replace("M7","").replace("7","").replace("M","")
    t=t.replace("MINOR","").replace("MAJOR","").replace(" ", "")
    for k in KEYS:
        if t.startswith(k): return k
    return None

def compute_avail_roots(rows: List[Dict]) -> Tuple[Counter, List[str]]:
    roots=[]
    for r in rows:
        if r.get("type") not in TONAL: continue
        k=str(r.get("key_guess","")).strip().replace("b","#")
        rt=_root_from_token(k)
        if rt: roots.append(rt)
    ctr=Counter(roots)
    order=[r for r,_ in ctr.most_common()]
    return ctr, order

def is_minor(ch:str)->bool:
    return ch.endswith("m") and not ch.endswith("dim")

def expand_with_sharps(base_pool: List[str], avail_ctr: Counter, roots_sorted: List[str],
                       prefer_minor: Optional[bool]=None, max_extra:int=6) -> List[str]:
    pool = list(dict.fromkeys(base_pool))
    if prefer_minor is None:
        if pool:
            minor_ratio = sum(is_minor(c) for c in pool)/len(pool)
            prefer_minor = (minor_ratio >= 0.5)
        else:
            prefer_minor = True
    used=set(ch.replace("m","").upper() for ch in pool)
    added=0
    for r in roots_sorted:
        if r in used: continue
        ch = (r+"m") if prefer_minor else r
        if avail_ctr.get(r,0)<=0: continue
        pool.append(ch); used.add(r); added+=1
        if added>=max_extra: break
    return pool[:max_extra] if len(pool)>max_extra else pool

# fair rotation chooser
CHORD_USE_COUNT = defaultdict(int)

def chord_root(ch:str)->Optional[str]:
    return ch.replace("m","").upper() if isinstance(ch,str) else None

def semitone_dist(a:str,b:str)->int:
    ra, rb = chord_root(a), chord_root(b)
    if not ra or not rb: return 0
    ia, ib = KEYS.index(ra), KEYS.index(rb)
    d=abs(ib-ia)
    return min(d, 12-d)

def pick_emotion_chord(emotion:str, prev_chord:str, EMO_TO_CHORDS:Dict[str,List[str]],
                       temp:float, mem:RecentMem) -> str:
    pool = EMO_TO_CHORDS.get(emotion, EMO_TO_CHORDS.get("neutral", ["Am","C","Em","Dm"]))
    if not pool: pool=["Am"]
    # score: fewer uses better; closer to prev better; penalize very recent
    recent=set(mem.buf) if mem else set()
    scores=[]
    for ch in pool:
        freq=CHORD_USE_COUNT[ch]
        dist=semitone_dist(prev_chord, ch)
        penal=2.0 if ch in recent else 0.0
        score = freq*1.2 + dist*0.8 + penal
        scores.append(score)
    weights=softmax_scores([-s for s in scores], t=max(1e-3, temp))
    u=random.random(); acc=0.0
    for ch, w in zip(pool, weights):
        acc+=w
        if u<=acc:
            CHORD_USE_COUNT[ch]+=1
            return ch
    CHORD_USE_COUNT[pool[0]]+=1
    return pool[0]
# <<< MOD END

# ========= Selection =========
def softmax_pick(cand: List[Dict], chord: str, energy: float,
                 allowed: set, k: int, temp: float, mem: RecentMem) -> List[Dict]:
    pool=[r for r in cand if r["type"] in allowed and r["path"].startswith(AUDIO_DIR)]
    if not pool or k<=0: return []
    scores=[]
    for r in pool:
        kd=abs(semitone_shift(r.get("key_guess","C"), chord))
        ed=abs(float(r.get("energy",0.5)) - energy)
        s=kd*2.0+ed if r["type"] in TONAL else ed+kd*0.2
        if r["path"] in mem.buf: s += 1.0
        scores.append(s)
    weights=softmax_scores(scores, t=temp)
    sel=[]
    idx=list(range(len(pool)))
    for _ in range(min(k,len(pool))):
        u=random.random(); acc=0.0; chosen=-1
        for j in range(len(idx)):
            acc+=weights[idx[j]]
            if u<=acc:
                chosen=idx.pop(j); break
        if chosen<0: chosen=idx.pop(0)
        sel.append(pool[chosen])
        # renormalize
        ssum=sum(weights[i] for i in idx) or 1.0
        weights=[w/ssum for w in weights]
    mem.push([r["path"] for r in sel])
    return sel

def choose_block_layers(pool: List[Dict], chord: str, energy: float, mem: RecentMem) -> Tuple[List[Dict], List[Dict]]:
    core=[]
    core += softmax_pick(pool, chord, energy, {"Kick","Drum","Clap"}, 1, TEMP, mem)
    core += softmax_pick(pool, chord, energy, {"Bass","Pad","Synth"}, 1, TEMP, mem)
    core += softmax_pick(pool, chord, energy, TONAL|CORE, 1, TEMP, mem)
    orn  = softmax_pick(pool, chord, energy, ORN, 1 if energy>=0.45 else 0, TEMP, mem)
    return core, orn

def build_block(chord: str, pool: List[Dict], target_energy: float, dur_ms: int, mem: RecentMem) -> AudioSegment:
    lp_pad = int(random.uniform(9000,12000) if VAR>0 else 10000)
    hp_drum= int(random.uniform(50,70) if VAR>0 else 60)
    hp_orn = int(random.uniform(3200,4800) if VAR>0 else 3500)

    core_sel, orn_sel = choose_block_layers(pool, chord, target_energy, mem)

    # log chosen layers (relative paths)
    def rel(p): 
        try: return os.path.relpath(p, AUDIO_DIR)
        except Exception: return p
    if core_sel:
        print("  🧩 CORE:", " | ".join([f"{r['type']}:{rel(r['path'])}" for r in core_sel]), flush=True)
    if orn_sel:
        print("  ✨ ORN :", " | ".join([f"{r['type']}:{rel(r['path'])}" for r in orn_sel]), flush=True)
    if not core_sel and not orn_sel:
        print("  ⚠️ No loops picked this block — dropping in silence.", flush=True)

    core=AudioSegment.silent(duration=dur_ms); first=True
    for r in core_sel:
        s=load_audio(r["path"])
        if s is None: continue
        if r["type"] in TONAL:
            s=pitch_shift(s, semitone_shift(r.get("key_guess","C"), chord))
            if r["type"] in {"Pad","Synth"}: s=low_pass_filter(s, lp_pad)
        else:
            s=high_pass_filter(s, hp_drum)
        s=s+random.uniform(-4,-1)
        try: s=s.pan(random.uniform(-0.35,0.35))
        except: pass
        s=tile_to(s, dur_ms)
        core = s if first else core.overlay(s); first=False

    orn=AudioSegment.silent(duration=dur_ms)
    for r in orn_sel:
        s=load_audio(r["path"])
        if s is None: continue
        s=high_pass_filter(s, hp_orn)
        s=s+random.uniform(-8,-4)
        try: s=s.pan(random.uniform(-0.5,0.5))
        except: pass
        s=tile_to(s, dur_ms)
        orn=orn.overlay(s)

    core=core.fade_in(80).fade_out(CROSSFADE_MS)
    full=core.overlay(orn).fade_in(80).fade_out(CROSSFADE_MS)
    return full

# ========= Camera: only vote inside the detection gate =========
emotion_q=queue.Queue()
stop_flag=threading.Event()
ui_lock=threading.Lock()
detect_gate=threading.Event()       # toggled by performer thread
gate_in_progress=threading.Event()  # avoid re-opening within a block

ui_state={"frame":None,"faces":[],"raw":"neutral","norm":"neutral","conf":0.0,"top3":[("neutral",1.0)]}

def decode_yolo(res, names):
    probs=getattr(res,"probs",None)
    if probs is not None:
        try:
            idx=int(probs.top1)
            conf=float(getattr(probs,"top1conf",0.0)) if hasattr(probs,"top1conf") else float(probs.data.max().item())
            vec=list(probs.data.cpu().numpy().ravel())
        except Exception:
            arr=getattr(probs,"data",None)
            if arr is None:
                idx=0; conf=0.0; vec=[1.0]
            else:
                arr=list(arr) if not hasattr(arr,"cpu") else list(arr.cpu().numpy().ravel())
                idx=arr.index(max(arr)); conf=max(arr); vec=arr
        label=names.get(idx,str(idx)) if isinstance(names,dict) else (names[idx] if 0<=idx<len(names) else str(idx))
        pairs=[(names.get(i,str(i)) if isinstance(names,dict) else names[i], float(v)) for i,v in enumerate(vec)]
        pairs.sort(key=lambda x:x[1], reverse=True)
        return label, conf, pairs[:3]
    boxes=getattr(res,"boxes",None)
    if boxes is not None and len(boxes)>0:
        i=int(boxes.conf.argmax()); cls=int(boxes.cls[i].item()); conf=float(boxes.conf[i].item())
        label=names.get(cls,str(cls)) if isinstance(names,dict) else (names[cls] if 0<=idx<len(names) else str(cls))
        return label, conf, [(label, conf)]
    return "neutral", 0.0, [("neutral",1.0)]

# >>> MOD START: normalize to 7 emotions
def normalize_label(raw: str) -> str:
    rl=(raw or "").lower()
    if "hap" in rl or "smil" in rl:     return "happy"
    if "surp" in rl:                    return "surprised"
    if "fear" in rl or "scare" in rl:   return "fear"
    if "ang"  in rl or "rage"  in rl:   return "angry"
    if "disgust" in rl or "revuls" in rl: return "disgust"
    if "sad" in rl or "frown" in rl:    return "sad"
    return "neutral"
# <<< MOD END

def camera_worker(model_path: str, cam_index: int=-1):
    try:
        from ultralytics import YOLO
        import cv2
        model=YOLO(model_path)
        print("✅ YOLO loaded:", model_path); print("🧾 classes:", model.names)
    except Exception as e:
        print("⚠️ YOLO unavailable:", e); return

    def open_cap(idx):
        import cv2
        for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            cap=cv2.VideoCapture(idx, backend)
            if cap.isOpened(): return cap
            cap.release()
        return None

    cap=None
    for i in ([cam_index] if cam_index>=0 else [0,1,2,3]):
        cap=open_cap(i)
        if cap is not None:
            print(f"🎥 Camera ready: index={i}")
            break
    if cap is None:
        print("⚠️ Couldn’t open camera"); return

    import cv2
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    local_hist=[]; gate_start=0.0

    try:
        while not stop_flag.is_set():
            ok,frame=cap.read()
            if not ok: time.sleep(0.02); continue
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray, 1.2, 5)
            crop=None
            if len(faces)>0:
                x,y,w,h=max(faces, key=lambda b:b[2]*b[3])
                pad=int(0.15*max(w,h))
                x0=max(0,x-pad); y0=max(0,y-pad)
                x1=min(frame.shape[1],x+w+pad); y1=min(frame.shape[0],y+h+pad)
                crop=frame[y0:y1,x0:x1]
            img=crop if crop is not None else frame

            try:
                res=model.predict(img, imgsz=256, conf=0.10, verbose=False)[0]
                raw,conf,top3=decode_yolo(res, model.names)
            except Exception:
                raw,conf,top3="neutral",0.0,[("neutral",1.0)]

            norm=normalize_label(raw)

            # UI state update (read by main thread)
            with ui_lock:
                ui_state.update(frame=frame, faces=faces, raw=raw, norm=norm, conf=float(conf), top3=top3)

            # Only vote when the gate is open
            if detect_gate.is_set():
                if not gate_in_progress.is_set():
                    gate_in_progress.set()
                    local_hist.clear()
                    gate_start=time.time()
                    print("🎯 Sampling emotion near block end …", flush=True)
                local_hist.append(norm)
                if (time.time()-gate_start) >= DETECT_WIN_SEC:
                    if local_hist:
                        winner=max(set(local_hist), key=local_hist.count)
                        try: emotion_q.put_nowait((winner, float(conf)))
                        except queue.Full: pass
                        print(f"🎯 Voted emotion → {winner}", flush=True)
                    detect_gate.clear()
                    gate_in_progress.clear()

            time.sleep(0.02)
    finally:
        cap.release()

# ========= Performer thread (gate → next block) =========
def performer_worker():
    try:
        pool = load_loops()
        print(f"🎼 Loops available: {len(pool)} (root: {AUDIO_DIR})", flush=True)

        beat=60.0/BPM; bar=beat*4; block_ms=int(round(bar*BARS_PER_BLOCK*1000))
        print(f"▶ Booting with chord={CALM_CHORD}  energy={DEFAULT_ENERGY:.2f}  block≈{block_ms/1000:.2f}s", flush=True)

        mem=RecentMem(NO_REPEAT_MEMORY)
        current_chord=CALM_CHORD
        desired_chord=CALM_CHORD
        energy_base=DEFAULT_ENERGY

        # >>> MOD START: build emotion→chord POOLS from actual library roots (includes sharps)
        avail_ctr, roots_sorted = compute_avail_roots(pool)
        EMO_TO_CHORDS = {}
        for emo, base in EMO_TO_CHORDS_BASE.items():
            EMO_TO_CHORDS[emo] = expand_with_sharps(base, avail_ctr, roots_sorted, prefer_minor=None, max_extra=6)
        print("🎛  Emotion→Chord pools:", {k: v for k,v in EMO_TO_CHORDS.items()}, flush=True)

        energy_preset = {
            "neutral":   0.60,
            "happy":     0.75,
            "surprised": 0.70,
            "sad":       0.65,
            "angry":     0.70,
            "fear":      0.55,
            "disgust":   0.50,
        }
        # <<< MOD END

        seg=build_block(current_chord, pool, energy_base, block_ms, mem)
        h_current=play_block(seg); t_start=time.time()
        built_next=None; h_next=None
        gate_armed=False
        gate_advance = PREBUILD_SEC + (CROSSFADE_MS/1000.0) + DETECT_WIN_SEC

        while not stop_flag.is_set():
            now=time.time(); elapsed=now-t_start
            # open the gate near the end so camera can vote
            if not gate_armed and elapsed >= (block_ms/1000.0 - gate_advance):
                detect_gate.set()
                gate_armed=True

            # consume gate result if any
            try:
                lbl,conf=emotion_q.get_nowait()
                # >>> MOD START: pick chord from pool (fair rotation), align energy
                desired_chord = pick_emotion_chord(lbl, current_chord, EMO_TO_CHORDS, TEMP, mem)
                energy_base   = energy_preset.get(lbl, DEFAULT_ENERGY)
                # <<< MOD END
                print(f"🪄 Emotion={lbl} → next_chord={desired_chord}  energy≈{energy_base:.2f}", flush=True)
            except queue.Empty:
                pass

            # prebuild next block (so we can crossfade)
            if built_next is None and elapsed >= (block_ms/1000.0 - PREBUILD_SEC - CROSSFADE_MS/1000.0):
                built_next=build_block(desired_chord, pool, energy_base, block_ms, mem)
                print(f"🔧 Prepared next: chord={desired_chord}", flush=True)

            # early start to crossfade
            if built_next and h_next is None and elapsed >= (block_ms/1000.0 - CROSSFADE_MS/1000.0):
                h_next=play_block(built_next)

            # block end: switch/retime/reset gate
            if elapsed >= (block_ms/1000.0 + 0.05):
                current_chord = desired_chord if built_next else current_chord
                h_current = h_next if h_next else h_current
                t_start=time.time()
                built_next=None; h_next=None
                gate_armed=False
                detect_gate.clear(); gate_in_progress.clear()
                print(f"▶ Now playing: {current_chord}", flush=True)

            time.sleep(0.02)

    except Exception as e:
        print("❌ Performer thread crashed:", repr(e), flush=True)
        import traceback; traceback.print_exc()

# ========= GUI (single place that uses imshow) =========
def ui_loop():
    import cv2, numpy as np
    title="Emotion (Q: quit | N/H/U/S/A/F/D: inject)"  # >>> MOD: update hint
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    while not stop_flag.is_set():
        key=cv2.waitKey(1)&0xFF; inject=None
        if key in (ord('q'),ord('Q')): stop_flag.set(); break
        # >>> MOD START: 7-class inject
        if key in (ord('n'),ord('N')): inject="neutral"
        if key in (ord('h'),ord('H')): inject="happy"
        if key in (ord('u'),ord('U')): inject="surprised"  # U = sUrprised
        if key in (ord('s'),ord('S')): inject="sad"
        if key in (ord('a'),ord('A')): inject="angry"
        if key in (ord('f'),ord('F')): inject="fear"
        if key in (ord('d'),ord('D')): inject="disgust"
        # <<< MOD END
        if inject is not None:
            try: emotion_q.put_nowait((inject,0.99))
            except queue.Full: pass
            print("⌨️ Inject →", inject, flush=True)

        with ui_lock:
            frame=None if ui_state["frame"] is None else ui_state["frame"].copy()
            faces=ui_state["faces"]; raw=ui_state["raw"]; norm=ui_state["norm"]; conf=ui_state["conf"]; top3=ui_state["top3"]

        if frame is None:
            frame=np.zeros((360,640,3),dtype=np.uint8)
            cv2.putText(frame,"Waiting for camera...",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)
        else:
            if len(faces)>0:
                x,y,w,h=max(faces, key=lambda b:b[2]*b[3]); 
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            txt=f"RAW:{raw}  EMO:{norm}  conf:{conf:.2f}"
            cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
            x0,y0=10,60
            for name,p in top3:
                bar_w=int(220*float(p))
                cv2.rectangle(frame,(x0,y0),(x0+bar_w,y0+14),(50,180,255),-1)
                cv2.putText(frame,f"{name}:{p:.2f}",(x0+230,y0+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); y0+=20

        cv2.imshow(title, frame)
    try: cv2.destroyAllWindows()
    except: pass

# ========= Main =========
def main():
    seed=int(time.time()*1000)%(2**31-1) if SEED is None else int(SEED)
    random.seed(seed)
    print(f"🎲 seed={seed} | default_chord={CALM_CHORD} | default_energy={DEFAULT_ENERGY:.2f}")
    print("📂 AUDIO_DIR =", AUDIO_DIR)
    audio_self_test()

    t_cam=threading.Thread(target=camera_worker, args=(MODEL_PATH,-1), daemon=True)
    t_perf=threading.Thread(target=performer_worker, daemon=True)
    t_cam.start(); t_perf.start()
    try:
        ui_loop()
    finally:
        stop_flag.set(); t_cam.join(timeout=1.0); t_perf.join(timeout=1.0)

if __name__=="__main__":
    if not os.path.isdir(AUDIO_DIR):
        print("❌ AUDIO_DIR doesn’t exist:", AUDIO_DIR); sys.exit(1)
    main()
