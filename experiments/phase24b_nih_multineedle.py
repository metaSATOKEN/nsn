# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 24b: Multi-needle NIH -- distractor-hardened variant
# ============================================================
# Phase 24's single-needle NIH saturated at 100% across all
# 60 (ctx, position, method) cells. The test was too easy on
# Llama-3.1-8B: a single distinctive token sequence inserted
# into a non-adversarial haystack is robustly retrievable
# even under lossy INT3 compression.
#
# Phase 24b hardens the protocol by embedding THREE needles
# with a structured label-to-code binding:
#
#     "The security token for Project Titan is XRQ7-BLUE-1984."
#     "The security token for Project Orion is LMN3-GREEN-2037."
#     "The security token for Project Vega  is PQZ5-AMBER-1776."
#
# The question specifies ONE project label and the model must
# produce THAT label's code, not one of the other two. A
# retrieval failure that corrupts the K-cache's ability to
# associate "Project X" with "code Y" will show up as a
# "wrong-code" response -- the model returns a code, but it's
# one of the two distractors rather than the target. This gives
# us a new metric beyond pass/fail:
#
#     pass         : target key appears in the 15-token decode
#     wrong-code   : a distractor's key appears (but not target)
#     hallucinate  : none of the three keys appear
#
# In a saturated regime, methods score identically. Under a
# genuine compression-induced degradation, we expect wrong-code
# rate to rise BEFORE pass rate collapses: cross-needle
# confusion is a softer failure mode than total retrieval loss.
#
# Protocol:
#   Model:        meta-llama/Llama-3.1-8B (same as Phase 24)
#   Haystack:     WikiText-2 test, repeated as needed
#   Needles:      3 project-labelled codes (above)
#   Positions:    fixed at pos fractions [0.2, 0.5, 0.8]; the
#                 three needles are permuted across these three
#                 slots each trial.
#   Trials:       10 trials per (ctx, method), covering 10
#                 (permutation, target-label) combinations.
#                 Fixed trial list for reproducibility.
#   Contexts:     8K, 32K
#   Methods:      FP16 / (C) v0.2 sym INT3 / (D) v0.3 asym INT3
#   Scoring:      3-way per trial (pass / wrong-code / hallucinate)
#                 based on substring match in 15-token greedy decode
#   Total runs:   2 ctx x 3 methods x 10 trials = 60
#
# Runtime estimate: ~25-30 min on Colab A100 (same order as
# Phase 24; the haystack is slightly longer due to 3 needles
# but the prefill cost is dominated by raw context length).
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
    """v0.2 symmetric nsep+pchan."""
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
    """v0.3 asymmetric nsep+pchan (the shipped recipe)."""
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
# === CELL V4 === Multi-needle specification + trial table
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

# 10 fixed (permutation, target_idx) trials per (ctx, method) cell.
# perm[i] is the index of the needle placed at POS_FRACS[i].
# target_idx is the needle we ask about in the question.
# This table ensures every target label appears at every position
# at least once across the 10 trials, so the aggregate measurement
# is not biased by a single position role.
TRIALS = [
    # perm       target_idx   label_at_asked_position
    ([0, 1, 2],  0),   # Titan at 0.2, question about Titan
    ([0, 1, 2],  1),   # Orion at 0.5, question about Orion
    ([0, 1, 2],  2),   # Vega  at 0.8, question about Vega
    ([2, 0, 1],  0),   # Titan at 0.5
    ([2, 0, 1],  1),   # Orion at 0.8
    ([2, 0, 1],  2),   # Vega  at 0.2
    ([1, 2, 0],  0),   # Titan at 0.8
    ([1, 2, 0],  1),   # Orion at 0.2
    ([1, 2, 0],  2),   # Vega  at 0.5
    ([0, 2, 1],  0),   # Titan at 0.2, Vega middle, Orion late  (distractor shuffle)
]


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext_full = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 full text: {len(wtext_full)} chars")


def build_multi_needle_prompt(ctx_target, perm, target_idx, tok):
    """Build a prompt of ~ctx_target tokens with 3 needles placed
    according to ``perm`` at positions POS_FRACS = [0.2, 0.5, 0.8],
    and a question asking for ``target_idx``'s label. Returns
    (prefix_ids, question_ids, target_needle_dict, slot_labels).
    """
    q_text = QUESTION_FMT.format(label=NEEDLES[target_idx]["label"])
    q_ids  = tok.encode(q_text, add_special_tokens=False)

    # Tokenize all three needle sentences (placed in perm order).
    needle_sents = [NEEDLE_FMT.format(**NEEDLES[perm[i]]) for i in range(3)]
    needle_ids_each = [tok.encode(s, add_special_tokens=False)
                       for s in needle_sents]
    total_needle_tokens = sum(len(n) for n in needle_ids_each)

    haystack_needed = ctx_target - len(q_ids) - total_needle_tokens
    if haystack_needed < 512:
        raise RuntimeError(f"ctx_target={ctx_target} too small for 3-needle protocol")

    # Tokenize enough haystack (repeat WikiText-2 if necessary)
    text = wtext_full
    while len(tok.encode(text[:haystack_needed * 6], add_special_tokens=False)) < haystack_needed + 200:
        text = text + "\n\n" + wtext_full
    hay_ids = tok.encode(text, add_special_tokens=False)
    hay_ids = hay_ids[:haystack_needed]

    # Compute absolute insertion indices in the (future) prefix.
    # Positions are expressed in units of haystack tokens.
    insert_positions = [int(pf * haystack_needed) for pf in POS_FRACS]
    # Enforce minimum separation from boundaries.
    insert_positions = [
        max(64, min(haystack_needed - 64, p)) for p in insert_positions
    ]
    # Sort ascending (they should already be; paranoia).
    ordered = sorted(zip(insert_positions, needle_ids_each))
    insert_positions = [p for p, _ in ordered]
    needle_ids_each  = [n for _, n in ordered]

    # Build the prefix by walking haystack and inserting needles at
    # the right token offsets.
    prefix_ids = []
    cursor = 0
    for pos, n_ids in zip(insert_positions, needle_ids_each):
        prefix_ids.extend(hay_ids[cursor:pos])
        prefix_ids.extend(n_ids)
        cursor = pos
    prefix_ids.extend(hay_ids[cursor:])

    # Trim / pad to exactly ctx_target - len(q_ids).
    target_prefix_len = ctx_target - len(q_ids)
    if len(prefix_ids) > target_prefix_len:
        prefix_ids = prefix_ids[:target_prefix_len]
    elif len(prefix_ids) < target_prefix_len:
        more = tok.encode(wtext_full, add_special_tokens=False)[:target_prefix_len - len(prefix_ids)]
        prefix_ids = prefix_ids + more

    # Build slot->label map for logging
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
    """Return one of 'pass', 'wrong-code', 'hallucinate'.

    pass        : target key present in decoded.
    wrong-code  : some other needle's key present (but not target).
    hallucinate : none of the three keys present.

    In the rare case where multiple keys appear, we count it as
    'pass' if target is among them (the model mentioned the right
    code even if it also mentioned distractors), otherwise
    'wrong-code'.
    """
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
# === CELL V6 === Evaluation loop
# =============================================================

METHODS = [
    ("FP16 baseline",    None),
    ("sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3)),
    ("asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3)),
]
CONTEXTS = [8192, 32768]

MODEL_ID = "meta-llama/Llama-3.1-8B"

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
print(f"Phase 24b multi-needle NIH summary "
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


out_path = "/content/phase24b_nih_multineedle.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":            "Multi-needle distractor NIH retrieval",
        "model":           MODEL_ID,
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
  HYPOTHESIS:
    Compression-induced K-cache error hurts label->code BINDING
    before it wipes out code retrieval entirely. We therefore
    expect 'wrong-code' to rise before 'pass' collapses under
    (C) or (D) relative to FP16, especially at 32K.

  WHAT TO LOOK FOR:
    1. 'pass' rate: does (D) match FP16? How does (C) compare?
       Previous Phase 24 was saturated; if 24b saturates too,
       Llama-3.1-8B is genuinely robust at 32K under INT3
       compression and we lock that down as our long-context
       claim.

    2. 'wrong-code' rate: this is the new signal. A non-zero
       wrong-code rate under (D) or (C) that does NOT appear
       under FP16 is DIRECT evidence that compression corrupts
       label-to-code binding. That would be the most informative
       outcome for the paper.

    3. Context scaling. If 8K saturates but 32K shows wrong-code
       under INT3, the long-context regime has a threshold we
       should report.

    4. sym vs asym at 32K. At ctx=1024 on Llama-3.1-8B the two
       were 3.44% vs 1.43% WikiText-2 dPPL. At 32K NIH, does
       the asymmetric recipe also win?

  OUTCOMES:
    still-saturated : both INT3 variants match FP16 at 100%
                      even under the harder protocol. The
                      paper's §5.7 reports this as a clean
                      generalization positive; the test is
                      still saturated and we acknowledge that.
    partial-miss    : 'wrong-code' rate under (C) or (D) rises
                      above FP16's rate. This is the discovery
                      we hoped to see: a softer failure mode
                      that WikiText-2 dPPL does not detect.
                      Paper §5.7 reports the finding; depending
                      on severity we tighten the scope of the
                      usability claim.
    hard-miss       : 'pass' rate under INT3 falls significantly
                      below FP16. Paper narrative shifts toward
                      a long-context-specific caveat.
""")
