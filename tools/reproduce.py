"""Reproduce packaged analysis on CPU; require a new output directory."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from verify_package import ROOT, verify
from source_layout import check as check_source_layout, expected_extraction, layout, safe_relative, source_file


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
        ("paper", paper / "code/build_tables_and_figures.py"),
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
            ROOT / "papers/loop-kv/data" / audit,
        )
        nll = json.loads((out / "numerical/nb05-nll-checks.json").read_text())
        report["raw_logit_nll"] = {
            "predictions": len(nll),
            "all_within_original_tolerance": all(x["error"] <= x["limit"] for x in nll),
        }
        loop = ROOT / "papers/loop-kv"
        loop_reference = ROOT / "data/validation/loop-initial-tables.json"
        # The original plotting script records its acquisition-time input names.
        # Project its eight verified inputs into a temporary tree with those names.
        loop_inputs = out / "plot-inputs"
        for identifier, digest in json.loads(loop_reference.read_text())["source_sha256"].items():
            source = source_file(ROOT, identifier)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("Figure input mismatch: " + identifier)
            target = loop_inputs / safe_relative(identifier)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
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
                "loop-figures",
                [sys.executable, str(loop / "code/build_paper_assets_v0_2.py"),
                 "--research-root", str(loop_inputs), "--output-dir", str(out / "loop-figures")],
            ),
            (
                "paper-data",
                [sys.executable, str(ROOT / "scripts/build_continuation_tables.py"),
                 "--research-root", str(ROOT), "--output-dir", str(out / "paper-data"),
                 "--integrated-base", str(out / "paper/initial-tables.json"),
                 "--loop-base", str(out / "loop-figures/paper-data-v0.2-2026-10-03.json")],
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
        for folder, name in [
            (paper, "paper-data-v0.5-2026-10-08.json"),
            (loop, "paper-data-v0.3-2026-10-08.json"),
        ]:
            for filename in (name, "numerical-audit-nb06-v0.1-2026-10-08.json"):
                rel = folder.relative_to(ROOT) / filename
                same(str(rel), out / "paper-data" / rel, folder / "data" / filename)
        same("initial looped tables", out / "loop-figures/paper-data-v0.2-2026-10-03.json", loop_reference)
        os.environ.update(
            {k: env[k] for k in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "MPLBACKEND"]}
        )
        import matplotlib.image as mpimg
        import numpy as np

        for folder, figure_dir in [
            (paper, out / "paper/figures"),
            (loop, out / "loop-figures/figures/v0.2"),
        ]:
            pairs = [(f, folder / "figures" / f.name) for f in sorted(figure_dir.glob("*.png"))]
            if len(pairs) != 5:
                raise ValueError("Expected five base figures for " + folder.name)
            pairs.append((out / "nb06/figures/nb06-results-v0.1-2026-10-08.png",
                          folder / "figures/figure-06-continuation.png"))
            for actual, frozen in pairs:
                report["rendering"].append({
                    "figure": str(frozen.relative_to(ROOT)),
                    "pixel_equal": bool(np.array_equal(mpimg.imread(actual), mpimg.imread(frozen))),
                    "svg_generated": actual.with_suffix(".svg").is_file(),
                })
        failed = (
            not all(x["passed"] for x in report["comparisons"])
            or extraction["failed_checks"] != 0
            or not report["raw_logit_nll"]["all_within_original_tolerance"]
            or report["nb06_nll_components_checked"] != 15420
            or report["raw_logit_nll"]["predictions"] != 432
            or len(report["rendering"]) != 12
            or not all(x["svg_generated"] for x in report["rendering"])
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
