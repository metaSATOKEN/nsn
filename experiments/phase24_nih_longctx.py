# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 24: Needle-in-a-Haystack long-context probe
# ============================================================
# The NSN paper v0.10 §6.3 Limitations explicitly flags:
#
#     "long-context retrieval (LongBench, needle-in-haystack)
#      are not touched"
#
# and the GenSpark MoA review (round 9) noted that the paper's
# main motivation -- long-context KV-cache memory pressure --
# contrasts with an evaluation context of 1024 tokens, two
# orders of magnitude smaller than the 128K+ deployments the
# §1 Introduction cites.
#
# This phase runs a scoped NIH probe on a single canonical
# model (Llama-3.1-8B, 128K native context) at two context
# lengths (8K and 32K), scoring whether the proposed v0.3
# asymmetric INT3 recipe preserves long-range factual recall
# compared to the FP16 baseline and the v0.2 symmetric recipe.
#
# Protocol:
#   Model:        meta-llama/Llama-3.1-8B
#   Haystack:     WikiText-2 test text, concatenated and repeated
#                 as needed to reach the target context length.
#                 (WikiText-2 was already the main-result corpus,
#                 so using it here removes any dataset-mismatch
#                 confound between this probe and §5.)
#   Needle:       "The special code mentioned here is XRQ7-BLUE-
#                 1984. Please remember it."  The key fragment
#                 "XRQ7" is a six-character alphanumeric chosen
#                 to be unambiguous and to tokenize the same way
#                 regardless of surrounding context.
#   Question:     "What is the special code mentioned in the
#                  text above?"
#   Contexts:     8K tokens, 32K tokens.
#   Needle positions: 10 uniform positions per context length,
#                 expressed as fractions 0.0, 0.1, ..., 0.9 of
#                 the haystack length (before question).
#   Methods:      FP16 baseline (no cache compression),
#                 (C) v0.2 sym nsep+pchan INT3,
#                 (D) v0.3 asym nsep+pchan INT3.
#                 We do not re-run (A) naive or (B) asym-no-nsep:
#                 both already fail at ctx=1024 on Llama-3.1-8B
#                 in the main benchmark, so rerunning at long
#                 context is not informative.
#   Scoring:      Greedy decode 15 tokens after the question.
#                 Pass = substring "XRQ7" appears in the decoded
#                 output. Partial credit is not awarded; the test
#                 is deliberately binary to match NIH convention.
#   Total runs:   2 ctx x 10 pos x 3 methods = 60 runs.
#
# Runtime estimate: ~25-30 min on Colab A100 (prefill at 32K
# dominates; ~15 s per 32K forward, ~3 s per 8K forward, plus
# ~15 min model load and haystack prep).
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
# === CELL V2 === Quantizers (inline; identical to the NSN paper §3.5)
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
# === CELL V4 === Haystack + needle construction
# =============================================================

NEEDLE_SENTENCE = (
    " The special code mentioned here is XRQ7-BLUE-1984. "
    "Please remember it. "
)
NEEDLE_KEY      = "XRQ7"   # substring used for scoring
QUESTION        = (
    "\n\nQuestion: What is the special code mentioned in the text above?\n"
    "Answer:"
)

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext_full = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 full text: {len(wtext_full)} chars")


def build_nih_prompt(ctx_target, pos_frac, tok):
    """Build a prompt of ~ctx_target tokens with the needle
    inserted at ``pos_frac`` of the haystack length. Returns the
    tokenized prefix (as a 1D LongTensor) and the tokenized question
    (as a 1D LongTensor). The prefix is scaled so that prefix + question
    equals ctx_target tokens.

    We do NOT include the answer continuation; decoding is done
    after the question is forwarded.
    """
    q_ids = tok.encode(QUESTION, add_special_tokens=False)
    n_ids = tok.encode(NEEDLE_SENTENCE, add_special_tokens=False)

    # How many filler tokens the haystack needs to provide.
    haystack_needed = ctx_target - len(q_ids) - len(n_ids)
    if haystack_needed < 256:
        raise RuntimeError(f"ctx_target={ctx_target} too small for this protocol")

    # Tokenize enough haystack; repeat WikiText-2 if we need more.
    text = wtext_full
    while len(tok.encode(text[:haystack_needed * 6], add_special_tokens=False)) < haystack_needed + 100:
        text = text + "\n\n" + wtext_full
    hay_ids = tok.encode(text, add_special_tokens=False)
    hay_ids = hay_ids[:haystack_needed]

    # Split haystack at position pos_frac, insert needle.
    pos_idx = int(pos_frac * haystack_needed)
    # Keep pos_idx away from exact boundary so needle is embedded, not appended
    pos_idx = max(64, min(haystack_needed - 64, pos_idx))
    prefix_ids = hay_ids[:pos_idx] + n_ids + hay_ids[pos_idx:]

    # Trim to exactly ctx_target - len(q_ids), so total prompt = ctx_target
    target_prefix_len = ctx_target - len(q_ids)
    if len(prefix_ids) > target_prefix_len:
        prefix_ids = prefix_ids[:target_prefix_len]
    elif len(prefix_ids) < target_prefix_len:
        # pad from haystack tail (shouldn't happen in practice)
        more = tok.encode(wtext_full, add_special_tokens=False)[:target_prefix_len - len(prefix_ids)]
        prefix_ids = prefix_ids + more

    return (torch.tensor(prefix_ids, dtype=torch.long),
            torch.tensor(q_ids, dtype=torch.long))


# =============================================================
# === CELL V5 === NIH single-run (compression between context and query)
# =============================================================

N_DECODE_TOKENS = 15


def nih_single_run(ctx_target, pos_frac, method_fn, mdl, tok):
    """Run one NIH trial. Returns (pass_bool, decoded_text)."""
    prefix_ids, q_ids = build_nih_prompt(ctx_target, pos_frac, tok)

    prefix = prefix_ids.unsqueeze(0).to("cuda")
    q      = q_ids.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        # (1) Forward the long haystack+needle prefix; populate cache.
        out_ctx = mdl(prefix, use_cache=True)
        past = out_ctx.past_key_values

        # (2) Compress the cache (this is the intervention under test).
        compress_cache(past, method_fn)

        # (3) Forward the question on top of the compressed cache.
        out_q = mdl(q, past_key_values=past, use_cache=True)
        past = out_q.past_key_values
        next_tok = out_q.logits[0, -1, :].argmax().item()

        # (4) Greedy decode N_DECODE_TOKENS more tokens, continuing
        #     to use the compressed-prefix cache. Newly generated
        #     tokens append uncompressed FP16 KV entries to the cache
        #     (standard behavior); only the haystack+needle+question
        #     portion is compressed.
        generated = [next_tok]
        for _ in range(N_DECODE_TOKENS - 1):
            inp = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
            out = mdl(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().item()
            generated.append(next_tok)

    decoded = tok.decode(generated, skip_special_tokens=True)
    passed = NEEDLE_KEY in decoded
    return passed, decoded


# =============================================================
# === CELL V6 === Evaluation loop
# =============================================================

METHODS = [
    ("FP16 baseline",    None),
    ("sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3)),
    ("asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3)),
]

CONTEXTS   = [8192, 32768]
POS_FRACS  = [round(i / 10, 2) for i in range(10)]   # 0.0, 0.1, ..., 0.9

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
print(f"Needle key: '{NEEDLE_KEY}'")
print(f"Needle sentence: '{NEEDLE_SENTENCE.strip()}'")

results = {}

for ctx in CONTEXTS:
    print(f"\n{'='*80}")
    print(f"Context = {ctx} tokens  ({ctx // 1024} K)")
    print(f"{'='*80}")

    for method_name, method_fn in METHODS:
        key = (ctx, method_name)
        print(f"\n  Method: {method_name}")
        pos_records = []
        n_pass = 0
        for pf in POS_FRACS:
            passed, decoded = nih_single_run(ctx, pf, method_fn, mdl, tok)
            pos_records.append({
                "pos_frac":     pf,
                "passed":       bool(passed),
                "decoded_15tk": decoded,
            })
            n_pass += int(passed)
            flag = "✓" if passed else "✗"
            short_decoded = decoded.replace("\n", " ")[:60]
            print(f"    pos={pf:.1f}  {flag}   \"{short_decoded}\"")

        acc = n_pass / len(POS_FRACS)
        print(f"    --> {n_pass}/{len(POS_FRACS)} passed  "
              f"(accuracy = {acc*100:.0f}%)")
        results[f"ctx{ctx}__{method_name}"] = {
            "ctx":        ctx,
            "method":     method_name,
            "n_passed":   n_pass,
            "n_total":    len(POS_FRACS),
            "accuracy":   acc,
            "per_position": pos_records,
        }


# =============================================================
# === CELL V7 === Summary table + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 24 NIH long-context summary "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)
print(f"  {'Context':<10}  {'Method':<22}  {'Passed':>10}  {'Accuracy':>10}  "
      f"{'Δ vs FP16':>12}")
print(f"  {'-'*10}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}")

for ctx in CONTEXTS:
    fp16_acc = results[f"ctx{ctx}__FP16 baseline"]["accuracy"]
    for method_name, _ in METHODS:
        r = results[f"ctx{ctx}__{method_name}"]
        acc = r["accuracy"]
        delta = (acc - fp16_acc) * 100
        dstr = f"{delta:+.1f} pp" if method_name != "FP16 baseline" else "  --   "
        print(f"  {ctx:>6} ({ctx//1024}K)  {method_name:<22}  "
              f"{r['n_passed']:>3}/{r['n_total']:<3}   "
              f"{acc*100:>8.1f}%   {dstr:>12}")


out_path = "/content/phase24_nih_longctx.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":             "Needle-in-a-Haystack retrieval",
        "model":            MODEL_ID,
        "haystack_source":  "WikiText-2 test split (repeated as needed)",
        "needle_sentence":  NEEDLE_SENTENCE.strip(),
        "needle_key":       NEEDLE_KEY,
        "question":         QUESTION.strip(),
        "contexts":         CONTEXTS,
        "pos_fracs":        POS_FRACS,
        "methods":          [m[0] for m in METHODS],
        "scoring":          f"substring '{NEEDLE_KEY}' in greedy-decode "
                            f"first {N_DECODE_TOKENS} tokens after question",
        "cache_split":      "prefix (compressed) + question (uncompressed) + "
                            "greedy-decode tail (uncompressed)",
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
  HYPOTHESIS (the NSN paper v0.10 §5 extrapolation):
      v0.3 asym INT3 at 3.141 bits/value preserves enough
      attention fidelity that long-range retrieval remains
      functional. The 5.09x compression measurement in §5.4
      is storage-only; this is the quality-at-long-context
      counterpart.

  WHAT TO LOOK FOR:
    1. Accuracy parity between (D) asym INT3 and FP16 baseline
       at 8K and 32K. If (D) ties FP16, the §5.4 extrapolation
       is validated end-to-end.

    2. Position dependence. If (D) fails uniformly across
       positions, the quantization error is uniform. If it
       fails preferentially at specific positions (e.g., early
       tokens in a long context), that suggests per-position
       error accumulation which would warrant a follow-up.

    3. (C) sym vs (D) asym gap at long context. At ctx=1024
       the two are close on Llama-3.1-8B (3.44% vs 1.43%).
       At 32K, does the gap widen, close, or reverse?

  OUTCOMES:
    match  : both INT3 variants match FP16 at 8K and 32K.
             §5 gets a "long-context generalization confirmed"
             paragraph + Appendix F with the position-vs-accuracy
             table.
    partial: (D) slightly underperforms (e.g., 1-2 position
             misses); Paper gets a graceful-degradation note.
    miss   : significant accuracy gap at 32K. DISCOVERY --
             the recipe has a long-context regime we did not
             anticipate. Paper needs narrative adjustment
             before arxiv submission. This is the most valuable
             possible outcome for the pre-submission question.
""")
