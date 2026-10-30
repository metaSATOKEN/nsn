"""Reproduce packaged analysis on CPU; require a new output directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_package import ROOT, verify
from source_layout import check as check_source_layout, expected_extraction, layout, public_markdown, public_document_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        ap.error("Output directory already exists; choose a new directory.")
    if out.is_relative_to(ROOT):
        ap.error("Choose an output directory outside the frozen repository.")
    package = verify(require_data=True)
    if package["status"] != "PASS":
        print(json.dumps(package, indent=2))
        return 1
    source_check = check_source_layout(ROOT)
    if source_check["status"] != "PASS":
        print(json.dumps(source_check, indent=2))
        return 1
    out.mkdir(parents=True)
    env = dict(
        os.environ,
        MPLCONFIGDIR=str(out / "plot-cache"),
        XDG_CACHE_HOME=str(out / "cache"),
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    report = {
        "package": package,
        "scope": "CPU saved-value and saved-logit checks; no model/GPU calls",
        "comparisons": [],
        "rendering": [],
        "environment_metadata": {},
        "status": "RUNNING",
        "source_layout": source_check,
        "removed_non_numerical_document_hash_checks": layout(ROOT)["removed_from_extraction_source_hashes"],
    }
    paper = ROOT / "papers/nsn-integrated"
    jobs = [
        ("extraction", paper / "code/extract_existing_evidence.py"),
        ("numerical", ROOT / "papers/loop-kv/code/extract_numerical_audit_v0_4.py"),
        ("paper", paper / "code/build_integrated_assets_v0_7.py"),
        ("english", paper / "code/build_integrated_english_v0_6.py"),
    ]
    try:
        for name, script in jobs:
            print("Running " + name, flush=True)
            command = [
                sys.executable,
                str(script),
                "--research-root",
                str(ROOT),
                "--output-dir",
                str(out / name),
            ]
            with (out / (name + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        def same(label, actual, frozen, summary=False):
            if label.startswith("extraction/"):
                x = json.loads(actual.read_text())
                y = expected_extraction(ROOT, actual.name, json.loads(frozen.read_text()))
                if summary:
                    report["environment_metadata"] = {
                        "current": x.pop("environment"),
                        "frozen": y.pop("environment"),
                    }
                ok = x == y
            else:
                ok = actual.read_bytes() == frozen.read_bytes()
            report["comparisons"].append({"item": label, "passed": ok})

        for f in sorted((out / "extraction").glob("*.json")):
            same(
                "extraction/" + f.name,
                f,
                paper / "data/extraction-v0.1" / f.name,
                summary=f.name == "summary.json",
            )
        extraction = json.loads((out / "extraction/summary.json").read_text())
        report["saved_value_checks"] = {
            "total": extraction["checks"],
            "failed": extraction["failed_checks"],
        }
        audit = "numerical-audit-v0.4-2026-10-03.json"
        same(
            "numerical/" + audit,
            out / "numerical" / audit,
            ROOT / "papers/loop-kv" / audit,
        )
        nll = json.loads((out / "numerical/nb05-nll-checks.json").read_text())
        report["raw_logit_nll"] = {
            "predictions": len(nll),
            "all_within_original_tolerance": all(x["error"] <= x["limit"] for x in nll),
        }
        for name in [
            "manuscript-ja-v0.7-2026-10-05.md",
            "paper-data-v0.4-2026-10-05.json",
        ]:
            same("paper/" + name, out / "paper" / name, paper / name)
        for name in [
            "manuscript-en-v0.6-2026-10-05.md",
            "english-build-manifest-v0.6.json",
        ]:
            same("english/" + name, out / "english" / name, paper / name)
        # Reuse the frozen continuation analyzer and supplement builder in fresh outputs.
        nb06_run = ROOT / "runs/loop-kv-nb06-20261008-075410-5621380a"
        extra_jobs = [
            (
                "nb06",
                [
                    sys.executable,
                    str(ROOT / "scripts/analyze_loop_kv_nb06_results.py"),
                    str(nb06_run),
                    str(out / "nb06"),
                ],
            ),
            (
                "supplement",
                [
                    sys.executable,
                    str(ROOT / "scripts/build_nb06_paper_supplements.py"),
                    "--research-root",
                    str(ROOT),
                    "--output-dir",
                    str(out / "supplement"),
                ],
            ),
        ]
        for name, command in extra_jobs:
            with (out / (name + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        nb_name = "nb06-results-v0.1-2026-10-08.json"
        same("nb06/" + nb_name, out / "nb06" / nb_name, ROOT / "reports" / nb_name)
        report["nb06_nll_components_checked"] = json.loads(
            (out / "nb06" / nb_name).read_text()
        )["nll_components_checked"]
        generated = json.loads(
            (out / "supplement/nb06-paper-build-manifest.json").read_text()
        )
        for rel in generated["outputs"]:
            actual = out / "supplement" / rel
            if rel.endswith(".md"):
                # Preserve frozen outputs; apply declared publication-only prose,
                # author metadata, names and navigation in a separate edition.
                public_rel = public_document_path(ROOT, rel)
                public = out / "public-edition" / public_rel
                public.parent.mkdir(parents=True, exist_ok=True)
                public.write_text(public_markdown(ROOT, rel, actual.read_text()))
                actual = public
            same("supplement/" + rel, actual, ROOT / public_document_path(ROOT, rel))
        report["current_manuscript_outputs"] = 4
        report["supplement_outputs_checked"] = len(generated["outputs"])
        os.environ.update(
            {k: env[k] for k in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "MPLBACKEND"]}
        )
        import matplotlib.image as mpimg
        import numpy as np

        for f in sorted((out / "paper/figures/v0.2").glob("*.png")):
            frozen = paper / "figures/v0.2" / f.name
            x = mpimg.imread(f)
            y = mpimg.imread(frozen)
            report["rendering"].append(
                {
                    "figure": f.name,
                    "pixel_equal": bool(x.shape == y.shape and np.array_equal(x, y)),
                    "svg_generated": f.with_suffix(".svg").is_file(),
                }
            )
        failed = (
            not all(x["passed"] for x in report["comparisons"])
            or extraction["failed_checks"] != 0
            or not report["raw_logit_nll"]["all_within_original_tolerance"]
            or len(report["rendering"]) != 5
        )
        render_diff = any(not x["pixel_equal"] for x in report["rendering"])
        report["status"] = (
            "FAIL"
            if failed
            else ("PASS_WITH_RENDERING_DIFFERENCES" if render_diff else "PASS")
        )
    except Exception as e:
        report["status"] = "FAIL"
        report["error"] = type(e).__name__ + ": " + str(e)
    (out / "reproduction-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
