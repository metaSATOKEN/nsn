# AC08: readable experiment distribution

Start with [the experiment design](../../plans/loop-kv-ac08-design-v0.1-2026-10-02.md), then `launch-settings.json`. All 47 files in `source/` are the exact expanded contents of the previously run OneCell, checked against its embedded SHA-256 manifest. No compressed executable payload is included in this entrypoint.

## Reading order

1. `source/loop_kv_ac08_inputs.json`: frozen materials, schedule and limits.
2. `source/loop_kv_ac08_runtime.py`: isolated environment and fixed dependencies.
3. `source/loop_kv_ac08.py`: model experiment orchestration.
4. Reconstruction/codec, attention/kernel and verifier modules in `source/`.
5. `run_colab.py`: readable launcher, deadlines, console handling and result ZIP recovery.

From the repository root, this checks visible source files only:

```sh
.venv/bin/python experiments/ac08/run_colab.py
```

For a separately provisioned Colab GPU session, make the repository files available, review the source and grant its `HF_TOKEN` Secret access. Then use Colab's Python to invoke `experiments/ac08/run_colab.py --run` from the repository root. This downloads dependencies/model files and executes the fixed experiment. Do not run it on a local Mac. No Drive mounting is required. The launcher retains the original environment constraints and limits; support for other environments is not implied.

The original OneCell and notebook stay in the author's local workflow. Historical cell builders and launcher fragments remain as readable source evidence, not as the public execution entrypoint. Base64 fields in codec-fixture JSON encode small binary **test data**, not executable Python. New readable-launcher GPU execution was not performed during packaging.

In Colab, after making this repository available and changing to its root, use the notebook kernel:

```python
%run experiments/ac08/run_colab.py
# After reviewing the source and limits, opt into actual GPU execution:
%run experiments/ac08/run_colab.py --run
```

Embedded `source/design.md` and `source/prereg.md` remain unchanged historical inputs. Their old relative links describe the original research layout; use the design link above for repository navigation.
