# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 22: MMLU 5-shot downstream task (single exploratory run)
# ============================================================
# The Paper B drafts through v0.5 have measured quality exclusively
# on WikiText-2 sliding-window perplexity (Phases 14, 16, 17). Both
# cross-review rounds flagged this as the paper's biggest scope
# limitation: ΔPPL is a proxy, and downstream-task accuracy is what
# readers actually care about for deployment.
#
# This phase is a one-shot exploratory check on downstream accuracy
# for v0.3 asymmetric INT3 (the shipped recipe), not a full multi-
# benchmark campaign. We answer a single question:
#
#     Does the proposed recipe preserve MMLU 5-shot accuracy on two
#     canonical benchmark models (Qwen2-7B, Llama-3.1-8B) relative to
#     the FP16 baseline?
#
# Protocol:
#   Task:          MMLU (Hendrycks et al., 2021), 5-shot, selected
#                  subjects only (100 questions each, 5 subjects).
#                  Total: 500 test questions per (model, method).
#   Subjects:      high_school_psychology, college_computer_science,
#                  global_facts, anatomy, moral_disputes -- a diverse
#                  slice across knowledge / reasoning / facts / ethics.
#   Models:        Qwen/Qwen2-7B, meta-llama/Llama-3.1-8B
#   Methods:       FP16 baseline, sym nsep+pchan INT3 (v0.2),
#                  asym nsep+pchan INT3 (v0.3)
#   Scoring:       Next-token argmax over {A, B, C, D} after the
#                  "Answer:" prompt. Cache compression is applied
#                  between the 5-shot prefix forward pass and the
#                  test-question forward pass, matching how the
#                  KV cache quantization would run at deployment.
#
# Runtime estimate: ~30-40 min on Colab A100 (6 full runs × 500
# questions × ~0.8 s each for the context+continuation split).
# ============================================================


# =============================================================
# === CELL V1 === Setup (fresh runtime recommended)
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
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizers (inline; same as Paper B §3.5)
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
# === CELL V4 === MMLU loader + 5-shot prompt formatting
# =============================================================

SUBJECTS = [
    "high_school_psychology",
    "college_computer_science",
    "global_facts",
    "anatomy",
    "moral_disputes",
]
QUESTIONS_PER_SUBJECT = 100   # cap; some subjects are smaller
N_SHOT = 5


def _format_question(ex, with_answer=False):
    """Standard MMLU question block."""
    q = ex["question"].strip()
    opts = ex["choices"]
    block = f"{q}\n"
    block += f"A. {opts[0].strip()}\n"
    block += f"B. {opts[1].strip()}\n"
    block += f"C. {opts[2].strip()}\n"
    block += f"D. {opts[3].strip()}\n"
    if with_answer:
        letter = ["A", "B", "C", "D"][ex["answer"]]
        block += f"Answer: {letter}"
    else:
        block += "Answer:"
    return block


def build_fewshot_prefix(subject_name, dev_split):
    """Return the 5-shot prefix string for a subject, using the
    first N_SHOT dev examples from the MMLU 'dev' split."""
    header = (f"The following are multiple choice questions (with "
              f"answers) about {subject_name.replace('_', ' ')}.\n\n")
    shots = []
    for i in range(min(N_SHOT, len(dev_split))):
        shots.append(_format_question(dev_split[i], with_answer=True))
    return header + "\n\n".join(shots) + "\n\n"


# =============================================================
# === CELL V5 === Scoring: next-token argmax over A/B/C/D
# =============================================================

def _letter_token_ids(tok):
    """Get the token IDs for ' A', ' B', ' C', ' D' (space-prefixed)
    as used after 'Answer:'. We require single-token, mutually
    distinct IDs for the scoring to be meaningful -- if any two
    letters share the same leading token (e.g., the tokenizer emits
    a stand-alone space token first), the argmax over letter logits
    is ambiguous. We assert both conditions and error out loudly
    rather than silently returning bad numbers; callers on such
    tokenizers should switch to multi-token choice log-likelihood
    scoring instead of this single-token shortcut.
    """
    ids = []
    multi_token_letters = []
    for letter in ["A", "B", "C", "D"]:
        toks = tok.encode(" " + letter, add_special_tokens=False)
        if len(toks) == 0:
            toks = tok.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            multi_token_letters.append((letter, toks))
        ids.append(toks[0])

    if multi_token_letters:
        raise RuntimeError(
            "Single-token scoring assumes each of ' A', ' B', ' C', ' D' "
            "tokenizes to exactly one token; got multi-token encodings "
            f"for: {multi_token_letters}. Switch to multi-token "
            "choice log-likelihood scoring on this tokenizer."
        )
    if len(set(ids)) != 4:
        raise RuntimeError(
            f"Single-token scoring requires four distinct letter token "
            f"IDs; got {ids} (duplicates). Argmax over letter logits "
            f"would be ambiguous; switch to multi-token scoring."
        )
    return ids


def score_one(prefix, question, mdl, tok, letter_ids, method_fn):
    """Apply cache compression between prefix and question, then
    return (pred_letter, gold_letter)."""
    prefix_ids = tok.encode(prefix, return_tensors="pt").to("cuda")
    question_ids = tok.encode(
        question, return_tensors="pt", add_special_tokens=False,
    ).to("cuda")

    with torch.inference_mode():
        # (1) Forward on the 5-shot prefix; populate the KV cache.
        out_ctx = mdl(prefix_ids, use_cache=True)
        past = out_ctx.past_key_values

        # (2) Compress the cache. This is where the quantization
        #     actually affects the downstream answer.
        compress_cache(past, method_fn)

        # (3) Forward on the test question with the compressed cache.
        out_tgt = mdl(
            question_ids, past_key_values=past, use_cache=False,
        )
        last_logits = out_tgt.logits[0, -1, :].float()

    letter_logits = torch.tensor([last_logits[i].item() for i in letter_ids])
    pred_idx = int(letter_logits.argmax())
    return pred_idx


def eval_subject(subject, mdl, tok, letter_ids, method_fn):
    """Returns (correct, total) for one subject."""
    dev = load_dataset("cais/mmlu", subject, split="dev")
    test = load_dataset("cais/mmlu", subject, split="test")
    n_eval = min(QUESTIONS_PER_SUBJECT, len(test))

    prefix = build_fewshot_prefix(subject, dev)

    correct = 0
    for i in range(n_eval):
        ex = test[i]
        q_block = _format_question(ex, with_answer=False)
        pred_idx = score_one(prefix, q_block, mdl, tok, letter_ids, method_fn)
        gold_idx = ex["answer"]
        if pred_idx == gold_idx:
            correct += 1

    return correct, n_eval


# =============================================================
# === CELL V6 === Per-model evaluation loop
# =============================================================

METHODS = [
    ("FP16 baseline",      None),
    ("sym INT3 (v0.2)",    lambda x: qa_nsep_pchan_sym(x, 3)),
    ("asym INT3 (v0.3)",   lambda x: qa_nsep_pchan_asym(x, 3)),
]


def run_model(model_id, label):
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    letter_ids = _letter_token_ids(tok)
    print(f"letter token ids: { {l: letter_ids[i] for i, l in enumerate('ABCD')} }")

    per_method = {}
    for method_name, method_fn in METHODS:
        print(f"\n  >>> Method: {method_name}")
        per_subject = {}
        total_correct = 0
        total_n = 0
        for subj in SUBJECTS:
            print(f"    {subj:>35s}: ...", end="", flush=True)
            correct, n = eval_subject(subj, mdl, tok, letter_ids, method_fn)
            acc = correct / n
            per_subject[subj] = {"correct": correct, "n": n, "accuracy": acc}
            total_correct += correct
            total_n += n
            print(f" {correct:3d}/{n:3d}  ({acc*100:5.1f}%)")
        overall = total_correct / total_n
        per_method[method_name] = {
            "per_subject": per_subject,
            "overall_correct": total_correct,
            "overall_n":       total_n,
            "overall_accuracy": overall,
        }
        print(f"    {'OVERALL':>35s}: {total_correct:3d}/{total_n:3d}  "
              f"({overall*100:5.1f}%)")

    return {"label": label, "model": model_id, "per_method": per_method}


# =============================================================
# === CELL V7 === Run on Qwen2-7B + Llama-3.1-8B
# =============================================================

qwen2_res = run_model("Qwen/Qwen2-7B", "Qwen2-7B (Mode 1)")
llama_res = run_model("meta-llama/Llama-3.1-8B", "Llama-3.1-8B (Mode 2)")


# =============================================================
# === CELL V8 === Summary table + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 22 MMLU 5-shot ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

print(f"  {'Model':<24}  {'Method':<22}  {'Accuracy':>10}  {'Δ vs FP16':>10}")
print(f"  {'-'*24}  {'-'*22}  {'-'*10}  {'-'*10}")
for res in [qwen2_res, llama_res]:
    fp16_acc = res["per_method"]["FP16 baseline"]["overall_accuracy"]
    for method_name in ["FP16 baseline", "sym INT3 (v0.2)", "asym INT3 (v0.3)"]:
        m = res["per_method"][method_name]
        acc = m["overall_accuracy"]
        delta = (acc - fp16_acc) * 100
        dstr = f"{delta:+.2f} pp" if method_name != "FP16 baseline" else " --    "
        print(f"  {res['label']:<24}  {method_name:<22}  {acc*100:>8.2f} %  {dstr:>10}")

out_path = "/content/phase22_mmlu_downstream.json"
combined = {
    "date":              datetime.now().isoformat(timespec='seconds'),
    "protocol":          {
        "task":          "MMLU 5-shot",
        "subjects":      SUBJECTS,
        "questions_per_subject_cap": QUESTIONS_PER_SUBJECT,
        "n_shot":        N_SHOT,
        "scoring":       "next-token argmax over A/B/C/D letter tokens",
        "cache_split":   "5-shot prefix (compressed) + test question",
    },
    "runs": {
        "Qwen2-7B":      qwen2_res,
        "Llama-3.1-8B":  llama_res,
    },
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  If (D) asym INT3 accuracy is within a few percentage points of FP16
  baseline on both models, the paper's WikiText-2 result generalizes
  to this particular downstream task.

  If (D) loses substantially on either model while WikiText-2 ΔPPL
  is small, the WikiText-2 metric is under-selling the downstream
  cost and we should treat the paper's 'usable' label more cautiously.

  Either direction is informative and gets reported honestly in the
  Paper B revision.
""")
