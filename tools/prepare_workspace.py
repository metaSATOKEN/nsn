"""Restore a new research workspace from the checked repository and local assets.

No download, model execution, or numerical recalculation. Never overwrites a directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from verify_package import (
    ROOT,
    INVENTORY,
    asset_manifest,
    safe_path,
    verify,
    verify_assets,
)


def prepare(root, assets_dir, destination, dry_run=False):
    root = root.resolve()
    assets_dir = assets_dir.resolve()
    # Reject an existing leaf, including dangling symlinks, before resolving it.
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists")
    destination = destination.resolve()
    if destination.is_relative_to(root) or destination.is_relative_to(assets_dir):
        raise ValueError(
            "Destination must be outside the repository and asset directory"
        )
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must exist")
    repo_check = verify(root)
    asset_check = verify_assets(root, assets_dir)
    if repo_check["status"] != "PASS" or asset_check["status"] != "PASS":
        raise ValueError(json.dumps({"repository": repo_check, "assets": asset_check}))
    rows = json.loads((root / INVENTORY).read_text())["files"]
    members = [m for a in asset_manifest(root) for m in a["members"]]
    plan = {
        "mode": "dry-run" if dry_run else "restore",
        "destination": str(destination),
        "repository_files": len(rows) + 1,
        "external_files": len(members),
        "uncompressed_external_bytes": sum(m["bytes"] for m in members),
        "assets": asset_check,
    }
    if dry_run:
        return plan
    destination.mkdir()
    # Copy only inventoried source files, never environments or hidden personal files.
    for row in rows + [{"path": INVENTORY}]:
        rel = safe_path(row["path"])
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    for a in asset_manifest(root):
        with zipfile.ZipFile(assets_dir / a["filename"]) as z:
            for m in a["members"]:
                target = destination / safe_path(m["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                count = 0
                with z.open(m["path"]) as source, target.open("xb") as output:
                    for b in iter(lambda: source.read(1024 * 1024), b""):
                        count += len(b)
                        if count > m["bytes"]:
                            raise ValueError("Oversized archive member")
                        h.update(b)
                        output.write(b)
                if count != m["bytes"] or h.hexdigest() != m["sha256"]:
                    raise ValueError("Restored hash mismatch: " + m["path"])
    result = verify(destination, require_data=True)
    if result["status"] != "PASS":
        raise ValueError(json.dumps(result))
    plan["verification"] = result
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-dir", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        result = prepare(ROOT, a.assets_dir, a.destination, a.dry_run)
    except (ValueError, OSError, zipfile.BadZipFile) as e:
        ap.exit(1, "ERROR: " + str(e) + "\n")
    print(json.dumps(result, indent=2))
