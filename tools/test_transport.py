"""Exercise restoration failure boundaries using small synthetic archives, not science data."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from prepare_workspace import prepare
from verify_package import INVENTORY, digest, safe_path, verify, verify_assets


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "repo"
        self.assets = self.base / "assets"
        (self.root / "metadata").mkdir(parents=True)
        self.assets.mkdir()
        (self.root / "README.md").write_text("fixture\n")
        payload = b"test data\x00\xff"
        self.member = {
            "path": "runs/sample/raw.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.zip = self.assets / "evidence.zip"
        with zipfile.ZipFile(self.zip, "w") as z:
            z.writestr(self.member["path"], payload)
        self.asset = {
            "filename": self.zip.name,
            "bytes": self.zip.stat().st_size,
            "sha256": digest(self.zip),
            "members": [self.member],
        }
        self.write_assets()
        self.seal()

    def tearDown(self):
        self.tmp.cleanup()

    def write_assets(self):
        (self.root / "metadata/release-assets.json").write_text(
            json.dumps({"assets": [self.asset]})
        )

    def seal(self):
        rows = [
            {
                "path": p.relative_to(self.root).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in self.root.rglob("*")
            if p.is_file() and p.relative_to(self.root).as_posix() != INVENTORY
        ]
        (self.root / INVENTORY).write_text(json.dumps({"files": rows}))

    def test_restore_and_exact_contents(self):
        dest = self.base / "new"
        r = prepare(self.root, self.assets, dest)
        self.assertEqual(r["verification"]["status"], "PASS")
        self.assertEqual(digest(dest / self.member["path"]), self.member["sha256"])

    def test_dry_run_has_no_destination(self):
        dest = self.base / "new"
        prepare(self.root, self.assets, dest, True)
        self.assertFalse(dest.exists())

    def test_existing_directory_refused(self):
        dest = self.base / "new"
        dest.mkdir()
        (dest / "keep").write_text("keep")
        with self.assertRaises(ValueError):
            prepare(self.root, self.assets, dest)
        self.assertEqual((dest / "keep").read_text(), "keep")

    def test_corrupt_asset_fails_before_write(self):
        self.zip.write_bytes(self.zip.read_bytes() + b"corrupt")
        dest = self.base / "new"
        with self.assertRaises(ValueError):
            prepare(self.root, self.assets, dest)
        self.assertFalse(dest.exists())

    def test_modified_source_fails(self):
        (self.root / "README.md").write_text("tampered")
        self.assertEqual(verify(self.root)["status"], "FAIL")

    def test_missing_asset_fails(self):
        self.zip.unlink()
        self.assertEqual(verify_assets(self.root, self.assets)["status"], "FAIL")

    def test_extra_archive_member_refused(self):
        with zipfile.ZipFile(self.zip, "a") as z:
            z.writestr("unexpected", "x")
        self.asset.update(bytes=self.zip.stat().st_size, sha256=digest(self.zip))
        self.write_assets()
        self.seal()
        with self.assertRaises(ValueError):
            prepare(self.root, self.assets, self.base / "new")

    def test_unsafe_paths_refused(self):
        for p in ["../escape", "/absolute", "a/../../x", "a\\b", "a//b", ""]:
            with self.subTest(path=p), self.assertRaises(ValueError):
                safe_path(p)

    def test_partial_workspace_detected(self):
        self.assertEqual(verify(self.root)["status"], "PASS")
        self.assertEqual(verify(self.root, require_data=True)["status"], "FAIL")

    def test_symlink_source_refused(self):
        target = self.base / "outside"
        target.write_text("fixture\n")
        (self.root / "README.md").unlink()
        (self.root / "README.md").symlink_to(target)
        self.assertEqual(verify(self.root)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
