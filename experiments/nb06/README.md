# NB06: readable experiment distribution

Start with [the experiment design](../../plans/loop-kv-nb06-design-v0.1-2026-10-08.md), then `launch-settings.json`. All 29 files in `source/` are the exact expanded contents of the previously run OneCell, checked against its embedded SHA-256 manifest. No compressed executable payload is included in this entrypoint.

## Kernel entry points

For implementation review, start with the [packed K/V GQA kernel](source/loop_kv_nb06_kernel.py): `pipeline()` selects a path, `read_v()` reconstructs V, `partial()` computes block results with query-to-KV-head indexing, and `merge()` combines them. The [model adapter](source/loop_kv_nb06_attention.py) connects single-token decoding; the [codec](source/loop_kv_nb06_codec.py) and [packed cache](source/loop_kv_nb06_packed.py) implement storage and updates. [Input checks](source/loop_kv_nb06_checks.py) and the [saved-output verifier](source/verify_loop_kv_nb06.py) cover validation. See the [kernel guide](../../README.md#gpu-kernels-where-to-start) for the Ouro and QK-component counterparts.

## Reading order

1. `source/loop_kv_nb06_inputs.json`: frozen materials, schedule and limits.
2. `source/loop_kv_nb06_runtime.py`: isolated environment and fixed dependencies.
3. `source/loop_kv_nb06.py`: model experiment orchestration.
4. Reconstruction/codec, attention/kernel and verifier modules in `source/`.
5. `run_colab.py`: readable launcher, deadlines, console handling and result ZIP recovery.

From the repository root, this checks visible source files only:

```sh
.venv/bin/python experiments/nb06/run_colab.py
```

For a separately provisioned Colab GPU session, make the repository files available, review the source and grant its `HF_TOKEN` Secret access. Then use Colab's Python to invoke `experiments/nb06/run_colab.py --run` from the repository root. This downloads dependencies/model files and executes the fixed experiment. Do not run it on a local Mac. No Drive mounting is required. The launcher retains the original environment constraints and limits; support for other environments is not implied.

The original OneCell and notebook stay in the author's local workflow. Historical cell builders and launcher fragments remain as readable source evidence, not as the public execution entrypoint. Base64 fields in codec-fixture JSON encode small binary **test data**, not executable Python. New readable-launcher GPU execution was not performed during packaging.

In Colab, after making this repository available and changing to its root, use the notebook kernel:

```python
%run experiments/nb06/run_colab.py
# After reviewing the source and limits, opt into actual GPU execution:
%run experiments/nb06/run_colab.py --run
```

Embedded `source/design.md` and `source/prereg.md` remain unchanged historical inputs. Their old relative links describe the original research layout; use the design link above for repository navigation.
