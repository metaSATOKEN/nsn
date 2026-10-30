"""Read-only repository and optional release-asset verification (standard library)."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "metadata/package-files.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(value):
    p = PurePosixPath(value)
    if (
        not value
        or p.is_absolute()
        or any(x in {"..", ""} for x in p.parts)
        or "\\" in value
        or p.as_posix() != value
    ):
        raise ValueError("Unsafe or noncanonical path: " + value)
    return p


def ignored(path):
    return (
        any(
            x
            in {
                ".git",
                ".venv",
                "__pycache__",
                ".cache",
                "outputs",
                "reproduced",
                ".ipynb_checkpoints",
            }
            for x in path.parts
        )
        or path.name == ".DS_Store"
    )


def asset_manifest(root=ROOT):
    return json.loads((root / "metadata/release-assets.json").read_text())["assets"]


def verify(root=ROOT, require_data=False):
    root = root.resolve()
    failures = []
    expected = json.loads((root / INVENTORY).read_text())["files"]
    external = [m for a in asset_manifest(root) for m in a["members"]]
    names = [x["path"] for x in expected + external]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate/overlapping manifest paths")
    checked = restored = 0
    for rows, required in [(expected, True), (external, require_data)]:
        for row in rows:
            rel = safe_path(row["path"])
            p = root / rel
            if not p.resolve().is_relative_to(root) or p.is_symlink():
                failures.append(
                    {"path": str(rel), "problem": "outside package or symlink"}
                )
                continue
            if not p.is_file():
                if required:
                    failures.append({"path": str(rel), "problem": "missing"})
                continue
            if p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
                failures.append({"path": str(rel), "problem": "modified"})
            checked += 1
            if rows is external:
                restored += 1
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink()) and not ignored(p.relative_to(root))
    }
    failures.extend(
        {"path": p, "problem": "unexpected"}
        for p in sorted(actual - set(names) - {INVENTORY})
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "repository_files": len(expected),
        "files_checked": checked,
        "external_files_present": restored,
        "external_files_total": len(external),
        "failures": failures,
    }


def verify_assets(root, assets_dir):
    failures = []
    checked = 0
    for a in asset_manifest(root):
        name = a["filename"]
        rel = safe_path(name)
        if len(rel.parts) != 1:
            raise ValueError("Asset filename must be a basename")
        p = assets_dir / rel
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(assets_dir.resolve())
            or not p.is_file()
        ):
            failures.append({"file": name, "problem": "missing or unsafe asset"})
            continue
        if p.stat().st_size != a["bytes"] or digest(p) != a["sha256"]:
            failures.append({"file": name, "problem": "size/hash mismatch"})
            continue
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
            expected = {m["path"]: m["bytes"] for m in a["members"]}
            if (
                len(infos) != len(expected)
                or {i.filename: i.file_size for i in infos} != expected
            ):
                failures.append({"file": name, "problem": "archive inventory mismatch"})
                continue
            for i in infos:
                safe_path(i.filename)
                if (i.external_attr >> 16) & 0o170000 == 0o120000 or i.is_dir():
                    raise ValueError("Unexpected directory or symlink in archive")
        checked += 1
    return {
        "status": "PASS" if not failures else "FAIL",
        "assets_checked": checked,
        "failures": failures,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--assets-dir", type=Path)
    ap.add_argument("--require-data", action="store_true")
    a = ap.parse_args()
    result = verify(a.root, a.require_data)
    if a.assets_dir:
        result["release_assets"] = verify_assets(a.root, a.assets_dir)
        if result["release_assets"]["status"] != "PASS":
            result["status"] = "FAIL"
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
