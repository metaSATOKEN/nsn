# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 24c: Multi-needle NIH on Qwen2-7B (n=1 -> n=2 validation)
# ============================================================
# Phase 24b ran the multi-needle distractor protocol on
# Llama-3.1-8B and observed a 20pp pass-rate gap between
# (D) asym INT3 (80% at 32K) and (C) sym INT3 (60% at 32K),
# with sym producing bit-mixing hallucinations like
# "3X-RED-1984" and "XRZ5-BLUE" that are not present in
# FP16 or asym outputs. At n=10 per cell, neither pairwise
# comparison reached conventional statistical significance,
# but the pattern and the qualitative hallucinations were
# suggestive.
#
# The obvious weakness of Phase 24b is n=1 in the model
# dimension: the entire long-context discrimination signal
# comes from a single Llama-3.1-8B run. Phase 24c re-runs
# the identical protocol on Qwen2-7B -- a different model
# family (Qwen), a different KV head count (4 vs Llama's 8),
# and a different training distribution -- to see whether
# the Phase 24b pattern holds or inverts. If asym > sym at
# 32K on both models, the discrimination signal is not a
# Llama-specific artifact; if it inverts or disappears,
# the mechanism is narrower than §5.3 currently claims.
#
# Qwen2-7B is the natural second-model pick because:
#   - it is the canonical Mode 1 model (INT4 valley) in the
#     main benchmark, so its long-context behavior has broader
#     narrative weight;
#   - it natively supports 32K context (the test ceiling);
#   - at 7B parameters it fits comfortably in A100 40GB even
#     with 32K prefill;
#   - its (C) sym vs (D) asym WikiText-2 dPPL gap at ctx=1024
#     is 4.80% vs 1.24% (Table 2), matching the direction
#     seen on Llama-3.1-8B.
#
# Protocol: IDENTICAL to Phase 24b (3 needles, 10 trials per
# cell, 8K and 32K contexts, FP16 / C / D methods, 3-way
# scoring on 15-token greedy decode). Only the model ID
# differs.
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
# === CELL V2 === Quantizers (inline; identical to Paper B §3.5)
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
# === CELL V3 === KV cache compression plumbing
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
# === CELL V4 === Multi-needle spec (identical to Phase 24b)
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
# === CELL V6 === Evaluation loop  (MODEL SWAP: Qwen2-7B)
# =============================================================

METHODS = [
    ("FP16 baseline",    None),
    ("sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3)),
    ("asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3)),
]
CONTEXTS = [8192, 32768]

MODEL_ID = "Qwen/Qwen2-7B"

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
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
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
# === CELL V7 === Summary table + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 24c multi-needle NIH summary  (Qwen2-7B)  "
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


out_path = "/content/phase24c_nih_multineedle_qwen2.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":            "Multi-needle distractor NIH retrieval (Qwen2-7B)",
        "model":           MODEL_ID,
        "note":            ("Phase 24c = Phase 24b protocol with MODEL_ID swapped "
                            "to Qwen/Qwen2-7B. All other parameters (needles, trials, "
                            "contexts, methods, scoring) are identical to 24b."),
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
  PHASE 24b (Llama-3.1-8B) RESULT for reference:
                        pass  wrong  halluc   Acc
       32K  FP16           9     1      0     90%
       32K  sym  (C)       6     2      2     60%   (<- bit-mix hallucinations)
       32K  asym (D)       8     2      0     80%

  WHAT TO LOOK FOR IN PHASE 24c:
    1. Does (D) asym INT3 hold better than (C) sym INT3 at 32K
       on Qwen2-7B too? A replication of the direction -- even
       if magnitudes differ -- defends against the "Llama-specific"
       critique.

    2. Are bit-mix hallucinations present under (C) on Qwen2-7B?
       Phase 24b showed "3X-RED-1984" and "XRZ5-BLUE" on Llama
       under sym INT3 but not under asym. If Qwen2-7B produces
       analogous cross-needle fusions under sym only, that is
       strong qualitative evidence independent of n.

    3. Is FP16 baseline on Qwen2-7B stronger or weaker than Llama's
       at 32K NIH? Qwen2-7B is the 'Mode 1' canonical (K-cache
       outlier-sensitive); its behavior at long context is itself
       interesting.

  OUTCOMES:
    replicate  : (D) asym > (C) sym at 32K on Qwen2-7B, with
                 or without statistical significance. Paper §5.7
                 reports both models side-by-side; n=1 critique
                 is blunted.
    invert     : (C) sym > (D) asym on Qwen2-7B. Surprising --
                 means the long-context discrimination signal
                 is Llama-specific. Paper scopes accordingly.
    saturate   : both methods match FP16 on Qwen2-7B. Then the
                 24b result is Llama-specific in the 'harder
                 task on a more retrieval-sensitive model' sense.
                 Paper notes this and discusses.
""")
