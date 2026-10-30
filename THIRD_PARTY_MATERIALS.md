# Third-party material and retained provenance

This is a navigation inventory of the material retained in this package. It does not relicense third-party contributions under the author's CC BY 4.0 license. Existing file headers and upstream terms remain applicable; model weights and the full corpus are not distributed.

| Material | Retained paths and provenance | Version / conditions recorded in this package |
| --- | --- | --- |
| Ouro model-specific sources | [AC08 inputs](experiments/ac08/source/loop_kv_ac08_inputs.json), sources and metadata in [AC08](experiments/ac08/source/) and its retained run | ByteDance/Ouro-1.4B, revision fcfcfbc429c7b10ee73dfc718033351d8936d2a8. Original source hashes and headers are retained; the author's license does not replace upstream terms. |
| Nanbeige model-specific sources | [Model file inventory](experiments/nb05/source/nb05_model_files.json), [source files](experiments/nb05/source/nb05_model_source/), inherited NB06 copies | Nanbeige/Nanbeige4.2-3B-Base, revision 4a38e817c14b9f2c4b69b1e05cbda90395872f6b. The included Python model/configuration files carry Apache-2.0 headers. Preserve those notices. |
| PG19 book list and metadata | [NB05 selection](experiments/nb05/source/loop_kv_nb05_candidates.json), corresponding AC08 selection and source directories | List/metadata revision 4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b. Book IDs, selection rules and upstream locations are recorded in the contracts. Dataset and individual-book terms apply. |
| Corpus-derived text fragments and token sequences | materials.json and material records under [runs](runs/); listed array evidence in companion ZIPs | Acquisition URLs, book IDs and content hashes are retained in each run. These are not newly authored text licensed by the NSN author; check the recorded source and its applicable conditions for the intended reuse. |
| Dependencies and inherited implementations | Existing headers, [requirements-offline.txt](requirements-offline.txt), experiment-specific fixed dependency records | Third-party dependencies retain their own licenses. Installed packages and model weights are not bundled here. |

See [LICENSES.md](LICENSES.md) for author-owned material, [NOTICE](NOTICE) for attribution, and [coverage](DATA_COVERAGE.md) for the retained records and omitted inputs. This inventory is based on saved source records and headers; it is not a new review of every upstream license.
