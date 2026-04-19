# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 24d: Multi-needle NIH on Qwen2.5-14B (3rd model; pool n=30)
# ============================================================
# Phase 24b (Llama-3.1-8B) and 24c (Qwen2-7B) both showed that
# asym INT3 (D) retains multi-needle retrieval better than
# sym INT3 (C) at 32K context, with bit-mixing hallucinations
# clustering on (C). Per-cell sample size of n=10 left each
# pairwise comparison individually underpowered; the
# 2-model signal is suggestive but not conclusive on its own.
#
# Phase 24d adds Qwen2.5-14B as a 3rd model and pools the
# three runs to n=30 per cell. Qwen2.5-14B is picked because:
#   - it is in the catastrophic-(B) cluster in the main
#     ablation (145.67 % of baseline PPL, Table 2) like
#     Llama-3.1-8B, extending the catastrophic-family
#     coverage to two of three tested 32K-native models;
#   - it has the highest L0 V-CV in our 8-model set
#     (0.602, Table 2b). If §5.3's V-CV -> failure mechanism
#     generalizes to long-context discrimination, Qwen2.5-14B
#     should show the *largest* sym-vs-asym gap at 32K;
#   - it is from the same family (Qwen) as Qwen2-7B but a
#     different parameter scale, which also tests intra-family
#     consistency of the long-context discrimination signal;
#   - it natively supports 128K context so 32K is well within
#     its training distribution.
#
# VRAM note: Qwen2.5-14B fp16 loads at ~29.5 GB on A100 40GB.
# Adding a 32K-token KV cache (48 layers x 8 KV heads x 32K
# x 128 dim x 2 for K/V x 2 bytes fp16) is ~6.3 GB of
# resident cache plus ~1-2 GB activations during prefill.
# Total peak ~36-37 GB -- tight but fits.
#
# Protocol: IDENTICAL to Phase 24b / 24c (3 project-labeled
# needles, 10 trials per cell via the fixed TRIALS table,
# 8K + 32K contexts, FP16 / (C) sym / (D) asym methods,
# 3-way scoring on 15-token greedy decode).
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import torch
try:
    del mdl, tok
except NameError:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer datasets
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizers
# =============================================================

def qa_nsep_pchan_sym(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits - 1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_asym(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=0, keepdim=True)
    d_max = d.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


# =============================================================
# === CELL V3 === KV cache plumbing
# =============================================================

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    else:
        past[li] = (k, v)

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def _layer_has_kv(past, li):
    if hasattr(past, 'layers'):
        layer = past.layers[li]
        return (hasattr(layer, 'keys') and hasattr(layer, 'values')
                and layer.keys is not None and layer.values is not None)
    return past[li] is not None and past[li][0] is not None

def compress_cache(past, method_fn):
    if method_fn is None:
        return
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)


# =============================================================
# === CELL V4 === Multi-needle spec (identical to 24b/24c)
# =============================================================

NEEDLES = [
    {"label": "Project Titan", "code": "XRQ7-BLUE-1984",  "key": "XRQ7"},
    {"label": "Project Orion", "code": "LMN3-GREEN-2037", "key": "LMN3"},
    {"label": "Project Vega",  "code": "PQZ5-AMBER-1776", "key": "PQZ5"},
]

NEEDLE_FMT   = " The security token for {label} is {code}. Keep this confidential. "
QUESTION_FMT = ("\n\nQuestion: What is the security token for {label}?\n"
                "Answer:")
POS_FRACS    = [0.2, 0.5, 0.8]

TRIALS = [
    ([0, 1, 2], 0),
    ([0, 1, 2], 1),
    ([0, 1, 2], 2),
    ([2, 0, 1], 0),
    ([2, 0, 1], 1),
    ([2, 0, 1], 2),
    ([1, 2, 0], 0),
    ([1, 2, 0], 1),
    ([1, 2, 0], 2),
    ([0, 2, 1], 0),
]

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext_full = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 full text: {len(wtext_full)} chars")


def build_multi_needle_prompt(ctx_target, perm, target_idx, tok):
    q_text = QUESTION_FMT.format(label=NEEDLES[target_idx]["label"])
    q_ids  = tok.encode(q_text, add_special_tokens=False)

    needle_sents = [NEEDLE_FMT.format(**NEEDLES[perm[i]]) for i in range(3)]
    needle_ids_each = [tok.encode(s, add_special_tokens=False)
                       for s in needle_sents]
    total_needle_tokens = sum(len(n) for n in needle_ids_each)

    haystack_needed = ctx_target - len(q_ids) - total_needle_tokens
    if haystack_needed < 512:
        raise RuntimeError(f"ctx_target={ctx_target} too small for 3-needle protocol")

    text = wtext_full
    while len(tok.encode(text[:haystack_needed * 6], add_special_tokens=False)) < haystack_needed + 200:
        text = text + "\n\n" + wtext_full
    hay_ids = tok.encode(text, add_special_tokens=False)
    hay_ids = hay_ids[:haystack_needed]

    insert_positions = [int(pf * haystack_needed) for pf in POS_FRACS]
    insert_positions = [
        max(64, min(haystack_needed - 64, p)) for p in insert_positions
    ]
    ordered = sorted(zip(insert_positions, needle_ids_each))
    insert_positions = [p for p, _ in ordered]
    needle_ids_each  = [n for _, n in ordered]

    prefix_ids = []
    cursor = 0
    for pos, n_ids in zip(insert_positions, needle_ids_each):
        prefix_ids.extend(hay_ids[cursor:pos])
        prefix_ids.extend(n_ids)
        cursor = pos
    prefix_ids.extend(hay_ids[cursor:])

    target_prefix_len = ctx_target - len(q_ids)
    if len(prefix_ids) > target_prefix_len:
        prefix_ids = prefix_ids[:target_prefix_len]
    elif len(prefix_ids) < target_prefix_len:
        more = tok.encode(wtext_full, add_special_tokens=False)[:target_prefix_len - len(prefix_ids)]
        prefix_ids = prefix_ids + more

    slot_labels = [NEEDLES[perm[i]]["label"] for i in range(3)]

    return (torch.tensor(prefix_ids, dtype=torch.long),
            torch.tensor(q_ids,       dtype=torch.long),
            NEEDLES[target_idx],
            slot_labels)


# =============================================================
# === CELL V5 === Single-trial run + 3-way scoring
# =============================================================

N_DECODE_TOKENS = 15


def classify_decoded(decoded, target_needle):
    target_key = target_needle["key"]
    other_keys = [n["key"] for n in NEEDLES if n["key"] != target_key]
    if target_key in decoded:
        return "pass"
    if any(k in decoded for k in other_keys):
        return "wrong-code"
    return "hallucinate"


def multi_needle_run(ctx_target, perm, target_idx, method_fn, mdl, tok):
    prefix_ids, q_ids, target_needle, slot_labels = \
        build_multi_needle_prompt(ctx_target, perm, target_idx, tok)

    prefix = prefix_ids.unsqueeze(0).to("cuda")
    q      = q_ids.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        out_ctx = mdl(prefix, use_cache=True)
        past = out_ctx.past_key_values

        compress_cache(past, method_fn)

        out_q = mdl(q, past_key_values=past, use_cache=True)
        past = out_q.past_key_values
        next_tok = out_q.logits[0, -1, :].argmax().item()

        generated = [next_tok]
        for _ in range(N_DECODE_TOKENS - 1):
            inp = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
            out = mdl(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().item()
            generated.append(next_tok)

    decoded = tok.decode(generated, skip_special_tokens=True)
    outcome = classify_decoded(decoded, target_needle)
    return outcome, decoded, target_needle, slot_labels


# =============================================================
# === CELL V6 === Evaluation loop (MODEL SWAP: Qwen2.5-14B)
# =============================================================

METHODS = [
    ("FP16 baseline",    None),
    ("sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3)),
    ("asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3)),
]
CONTEXTS = [8192, 32768]

MODEL_ID = "Qwen/Qwen2.5-14B"

print(f"\nLoading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)
mdl.eval()
print(f"VRAM used after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")
for n in NEEDLES:
    print(f"  Needle: '{n['label']}' -> code='{n['code']}' key='{n['key']}'")

results = {}

for ctx in CONTEXTS:
    print(f"\n{'='*80}")
    print(f"Context = {ctx} tokens  ({ctx // 1024} K)")
    print(f"{'='*80}")

    for method_name, method_fn in METHODS:
        print(f"\n  Method: {method_name}")
        trial_records = []
        n_pass = n_wrong = n_halluc = 0
        for perm, target_idx in TRIALS:
            outcome, decoded, tgt, slot_labels = multi_needle_run(
                ctx, perm, target_idx, method_fn, mdl, tok
            )
            trial_records.append({
                "perm":         perm,
                "slot_labels":  slot_labels,
                "target_idx":   target_idx,
                "target_label": tgt["label"],
                "target_key":   tgt["key"],
                "outcome":      outcome,
                "decoded_15tk": decoded,
            })
            if   outcome == "pass":        n_pass   += 1
            elif outcome == "wrong-code":  n_wrong  += 1
            else:                          n_halluc += 1
            flag = {"pass": "✓", "wrong-code": "✗w", "hallucinate": "✗h"}[outcome]
            short_decoded = decoded.replace("\n", " ")[:50]
            print(f"    target={tgt['label']:<14s} perm={perm}  "
                  f"{flag}  \"{short_decoded}\"")

        total = len(TRIALS)
        acc = n_pass / total
        print(f"    --> pass {n_pass}/{total}  "
              f"wrong {n_wrong}/{total}  "
              f"halluc {n_halluc}/{total}   "
              f"accuracy = {acc*100:.0f}%")
        results[f"ctx{ctx}__{method_name}"] = {
            "ctx":        ctx,
            "method":     method_name,
            "n_pass":     n_pass,
            "n_wrong":    n_wrong,
            "n_halluc":   n_halluc,
            "n_total":    total,
            "accuracy":   acc,
            "trials":     trial_records,
        }


# =============================================================
# === CELL V7 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 24d multi-needle NIH summary  (Qwen2.5-14B)  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)
print(f"  {'Context':<10}  {'Method':<22}  {'pass':>6}  {'wrong':>6}  {'halluc':>6}  "
      f"{'Accuracy':>10}  {'Δ vs FP16':>10}")
print(f"  {'-'*10}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

for ctx in CONTEXTS:
    fp16_acc = results[f"ctx{ctx}__FP16 baseline"]["accuracy"]
    for method_name, _ in METHODS:
        r = results[f"ctx{ctx}__{method_name}"]
        acc = r["accuracy"]
        delta = (acc - fp16_acc) * 100
        dstr = f"{delta:+.1f} pp" if method_name != "FP16 baseline" else "  --   "
        print(f"  {ctx:>6} ({ctx//1024}K)  {method_name:<22}  "
              f"{r['n_pass']:>3}/{r['n_total']:<2}  "
              f"{r['n_wrong']:>3}/{r['n_total']:<2}  "
              f"{r['n_halluc']:>3}/{r['n_total']:<2}  "
              f"{acc*100:>8.1f}%   {dstr:>10}")


out_path = "/content/phase24d_nih_multineedle_qwen25_14b.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":            "Multi-needle distractor NIH retrieval (Qwen2.5-14B)",
        "model":           MODEL_ID,
        "note":            ("Phase 24d = Phase 24b/c protocol with MODEL_ID swapped "
                            "to Qwen/Qwen2.5-14B. All other parameters (needles, "
                            "trials, contexts, methods, scoring) are identical to "
                            "24b and 24c. Designed for pooled n=30 analysis."),
        "haystack_source": "WikiText-2 test split (repeated as needed)",
        "n_needles":       len(NEEDLES),
        "needles":         NEEDLES,
        "needle_format":   NEEDLE_FMT.strip(),
        "question_format": QUESTION_FMT.strip(),
        "pos_fracs":       POS_FRACS,
        "trials_per_cell": len(TRIALS),
        "contexts":        CONTEXTS,
        "methods":         [m[0] for m in METHODS],
        "scoring":         ("3-way per trial on 15-token greedy decode: "
                            "'pass' if target key present; 'wrong-code' "
                            "if a distractor key present but not target; "
                            "'hallucinate' if none of the three keys present"),
    },
    "results": results,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  PRIOR RESULTS (Llama-3.1-8B + Qwen2-7B):

  model          ctx  FP16  sym   asym    Δ(asym-sym)
  Llama-3.1-8B    8K  100%  90%   90%     +0 pp
  Llama-3.1-8B   32K   90%  60%   80%    +20 pp
  Qwen2-7B        8K  100%  50%   40%    -10 pp
  Qwen2-7B       32K  100%  30%   70%    +40 pp

  PREDICTIONS FOR QWEN2.5-14B:
    1. §5.3 mechanism predicts the sym-vs-asym gap at 32K
       scales with L0 V-CV. Qwen2.5-14B's L0 V-CV = 0.602
       is the HIGHEST in the 8-model set. If that prediction
       holds, we should see the largest (asym > sym) gap here.

    2. Cross-family consistency check: two Qwen models +
       one Llama = test whether the pattern is specific to
       a training pipeline or genuinely architecture-driven.

    3. Qwen2-7B's 8K asym-under-sym anomaly was speculated
       to link to short-T per-channel (min, max) estimation
       noise on a Mode-1 (high K-CV) model. Qwen2.5-14B is
       NOT Mode 1 (its INT4 sweep was monotone in Phase 1-4),
       so if the 8K asym-anomaly is Mode-1-specific we
       expect Qwen2.5-14B 8K to look more like Llama
       (tied at high accuracy) than like Qwen2-7B (asym
       < sym). If Qwen2.5-14B 8K ALSO shows asym < sym,
       the anomaly is Qwen-family-wide, not Mode-1-specific.

  POOLED ANALYSIS (after this run):
    3 models x 10 trials x 2 ctx x 3 methods = 180 total
    trials. Pooled at 32K across 3 models:
      sym:  n=30 cells (Llama 6 + Qwen2 3 + Qwen2.5 ?)
      asym: n=30 cells (Llama 8 + Qwen2 7 + Qwen2.5 ?)
    Fisher exact at pooled n=30 should reach significance
    if the direction is genuine.
""")
