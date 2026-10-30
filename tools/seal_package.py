"""Explicitly refresh the publication inventory after reviewed package edits.

Does not validate scientific results. The new inventory must be reviewed and versioned.
"""

import json
from pathlib import Path
from verify_package import ROOT, INVENTORY, digest, ignored, asset_manifest

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-reviewed-changes", action="store_true", required=True)
    a = p.parse_args()
    external = {m["path"] for z in asset_manifest(ROOT) for m in z["members"]}
    rows = []
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if (
            not f.is_file()
            or ignored(rel)
            or str(rel) == INVENTORY
            or str(rel) in external
        ):
            continue
        if f.is_symlink():
            raise ValueError("Symlinks are not distributed")
        rows.append(
            {"path": rel.as_posix(), "bytes": f.stat().st_size, "sha256": digest(f)}
        )
    (ROOT / INVENTORY).write_text(
        json.dumps({"version": 1, "files": rows}, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"repository_files": len(rows), "inventory": INVENTORY}))
