"""Check visible source loading without importing any experiment/model module."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ReadableSources(unittest.TestCase):
    def test_each_distribution(self):
        for key, count in [("ac08", 47), ("nb05", 31), ("nb06", 29)]:
            with self.subTest(experiment=key):
                entry = ROOT / "experiments" / key / "run_colab.py"
                spec = importlib.util.spec_from_file_location("public_launcher_" + key, entry)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(len(module.read_checked_sources()), count)
                with tempfile.TemporaryDirectory() as temp:
                    dest = Path(temp)
                    shutil.copytree(entry.parent / "source", dest / "source")
                    module.HERE = dest
                    target = dest / "source" / next(iter(module.EMBEDDED_SHA256))
                    original = target.read_bytes()
                    target.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "digest mismatch"):
                        module.read_checked_sources()
                    target.write_bytes(original)
                    (dest / "source" / "extra.py").write_text("raise RuntimeError('must not execute')")
                    with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                        module.read_checked_sources()

if __name__ == "__main__":
    unittest.main(verbosity=2)
