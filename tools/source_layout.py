"""Resolve acquisition-time source identifiers in the integrated publication layout.

Original result bytes and their identifiers are preserved. This explicit mapping
changes file lookup, not quantization, statistics, tolerances, or experiment logic.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=8)
def layout(root):
    return json.loads((Path(root) / "metadata/source-layout.json").read_text())

def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or p.as_posix() != value:
        raise ValueError("Unsafe source path: " + value)
    return p

def source_file(root, identifier):
    root = Path(root).resolve()
    record = layout(root)["files"].get(identifier)
    if identifier in layout(root)["excluded_documents"]:
        raise FileNotFoundError("Historical manuscript excluded from public package: " + identifier)
    rel = record["path"] if record else identifier
    p = root / safe_relative(rel)
    if p.is_symlink() or not p.resolve().is_relative_to(root):
        raise ValueError("Source path escapes repository")
    return p

def original_result_ids(root, prefix):
    return sorted(key for key in layout(Path(root).resolve())["files"]
                  if key.startswith(prefix + "/") and key.endswith(".json"))

def omitted_document(root, identifier):
    return identifier in layout(Path(root).resolve())["excluded_documents"]

def expected_extraction(root, filename, baseline):
    """Only two former document-presence/hash checks are removed from comparison.

    All scientific checks and every numeric value are otherwise compared exactly.
    The frozen baseline stays unchanged and the adjustment is explicit in reports.
    """
    excluded = layout(Path(root).resolve())["removed_from_extraction_source_hashes"]
    expected = json.loads(json.dumps(baseline))
    if filename == "source-manifest.json":
        for identifier in excluded:
            expected.pop(identifier)
    elif filename == "checks.json":
        labels = {"source unchanged " + identifier for identifier in excluded}
        removed = [row for row in expected if row["check"] in labels]
        if len(removed) != len(excluded) or not all(row["passed"] for row in removed):
            raise ValueError("Unexpected historical document-check baseline")
        expected = [row for row in expected if row["check"] not in labels]
    elif filename == "summary.json":
        expected["checks"] -= len(excluded)
    return expected

def public_document_path(root, relative_document):
    spec = layout(Path(root).resolve())
    names = dict(spec.get("publication_document_names", {}))
    names.update(spec.get("publication_navigation_copies", {}))
    return names.get(relative_document, relative_document)

def public_markdown(root, relative_document, text):
    """Apply declared publication prose, names and local-link presentation only."""
    spec = layout(Path(root).resolve())
    source_parent = posixpath.dirname(relative_document)
    target_parent = posixpath.dirname(public_document_path(root, relative_document))
    for before, after in spec.get("publication_text_replacements", {}).get(relative_document, []):
        if before in text:
            if text.count(before) != 1:
                raise ValueError("Ambiguous publication-only text replacement")
            text = text.replace(before, after)
        elif after not in text:
            raise ValueError("Missing publication-only text replacement anchor")
    def replace(match):
        target = match.group(1)
        path, marker, anchor = target.partition("#")
        if not path or re.match(r"\w+:", path):
            return match.group(0)
        original = posixpath.normpath(posixpath.join(source_parent, path))
        mapped = original
        if original in spec.get("publication_link_targets", {}):
            mapped = spec["publication_link_targets"][original]
        elif original in spec["files"]:
            mapped = spec["files"][original]["path"]
        elif original in spec.get("directories", {}):
            mapped = spec["directories"][original] + "/"
        elif original in spec["excluded_documents"]:
            url = spec["excluded_documents"][original].get("replacement_url")
            if url is None:
                raise ValueError("No replacement for excluded-document link: " + original)
            return "](" + url + (marker + anchor if marker else "") + ")"
        mapped_path, mapped_marker, mapped_anchor = mapped.partition("#")
        mapped_path = public_document_path(root, mapped_path)
        new = posixpath.relpath(mapped_path, target_parent)
        if mapped_path.endswith("/") or path.endswith("/"):
            new += "/"
        return "](" + new + (mapped_marker + mapped_anchor if mapped_marker else (marker + anchor if marker else "")) + ")"
    return re.sub(r"\]\(([^)]+)\)", replace, text)

def check(root):
    root = Path(root).resolve()
    failures = []
    for identifier, record in layout(root)["files"].items():
        p = source_file(root, identifier)
        if not p.is_file() or p.stat().st_size != record["bytes"]:
            failures.append(identifier)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(identifier)
    return {"status": "FAIL" if failures else "PASS",
            "source_files": len(layout(root)["files"]), "failures": failures}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", help="Original source identifier from a result/provenance record")
    args = parser.parse_args()
    if args.resolve:
        print(source_file(ROOT, args.resolve))
    else:
        result = check(ROOT)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 1)
