"""Test publication-only naming and link translation; no research data or models."""
import json
from pathlib import Path
import tempfile
import unittest
from source_layout import public_document_path, public_markdown, source_file

class PublicationEditionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'metadata').mkdir()
        self.spec = {
            'files': {'legacy/result.json': {'path': 'data/result.json'}},
            'directories': {'legacy/audit': 'data/validation'},
            'excluded_documents': {'private/note.md': {'reason': 'not distributed'}},
            'publication_document_names': {'papers/old.md': 'papers/NSN-2026-10-29-EN.md'},
            'publication_navigation_copies': {'plans/design.md': 'plans/public/design.md'},
            'publication_link_targets': {'colab/note.md': 'plans/README.md#colab'},
            'publication_text_replacements': {'papers/old.md': [['License pending.', 'Licensed under CC BY 4.0.']]},
        }
        (self.root/'metadata/source-layout.json').write_text(json.dumps(self.spec))

    def test_named_output_and_reference(self):
        self.assertEqual(public_document_path(self.root, 'papers/old.md'), 'papers/NSN-2026-10-29-EN.md')
        self.assertEqual(public_markdown(self.root, 'README.md', '[paper](papers/old.md)'), '[paper](papers/NSN-2026-10-29-EN.md)')

    def test_folder_move_keeps_correct_relative_target(self):
        self.assertEqual(public_markdown(self.root, 'plans/design.md', '[result](../legacy/result.json)'), '[result](../../data/result.json)')

    def test_directory_slash_and_anchor(self):
        raw='[audit](../legacy/audit#checks)'
        public='[audit](../data/validation/#checks)'
        self.assertEqual(public_markdown(self.root, 'papers/example.md', raw), public)
        self.assertEqual(public_markdown(self.root, 'papers/example.md', public), public)

    def test_declared_navigation_anchor(self):
        self.assertEqual(public_markdown(self.root, 'plans/design.md', '[guide](../colab/note.md)'), '[guide](../README.md#colab)')

    def test_external_and_fragment_unchanged(self):
        text='[web](https://example.org/paper#section) [local](#section)'
        self.assertEqual(public_markdown(self.root, 'plans/design.md', text), text)

    def test_explicit_prose_and_anchor_guard(self):
        self.assertEqual(public_markdown(self.root, 'papers/old.md', 'License pending.'), 'Licensed under CC BY 4.0.')
        with self.assertRaises(ValueError):
            public_markdown(self.root, 'papers/old.md', 'Unexpected prose.')
        with self.assertRaises(ValueError):
            public_markdown(self.root, 'papers/old.md', 'License pending. License pending.')

    def test_unmapped_private_document_is_rejected(self):
        with self.assertRaises(ValueError):
            public_markdown(self.root, 'papers/example.md', '[note](../private/note.md)')

    def test_numerical_source_lookup_is_separate(self):
        self.assertEqual(source_file(self.root, 'legacy/result.json'), self.root/'data/result.json')
        self.assertEqual(source_file(self.root, 'papers/old.md'), self.root/'papers/old.md')

if __name__ == '__main__':
    unittest.main()
