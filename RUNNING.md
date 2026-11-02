# Running the NSN publication package

Updated 29 October 2026. Run commands **from the research root using that root's `.venv/bin/python`**. Do not rely on activation or `PATH`; package installation uses `.venv/bin/python -m pip`.

## Within the original Looped Transformer research workspace

Stay beside its root `RUNNING.md` and `.venv/`:

```sh
.venv/bin/python --version
.venv/bin/python -B Github公開用/github-release/tools/verify_package.py
.venv/bin/python -B Github公開用/github-release/tools/verify_package.py --assets-dir Github公開用/github-release-assets
```

That environment was prepared with Python 3.13.17 and only pip. Scientific packages were not installed into it by publication preparation.

## Standalone repository or restored working copy

Use that copy's own root environment. If it is absent, first choose an installed Python by absolute path and create a new `.venv`; preserve any existing environment. The hash checker and restorer need only the standard library (Python 3.10+).

For a new copy with no .venv, replace the example absolute path with your selected installed Python executable:

~~~sh
/absolute/path/to/python3.13 --version
test ! -e .venv && /absolute/path/to/python3.13 -m venv .venv
.venv/bin/python -B -c 'import sys; print(sys.executable); print(sys.version)'
~~~

The guard stops if .venv already exists. Do not delete an existing environment to make the example run. File checks were verified locally with Python 3.13.17 on macOS. The fixed offline analysis dependencies were originally checked with Python 3.13.5; record the OS and resolved versions in your separate analysis copy. A matching Python name alone does not establish numerical reproducibility.


```sh
.venv/bin/python --version
.venv/bin/python -c 'import sys; print(sys.executable)'
.venv/bin/python -m pip --version
.venv/bin/python -B tools/verify_package.py
.venv/bin/python -B tools/verify_package.py --assets-dir ../github-release-assets
```

These commands read files only. `--assets-dir` checks ZIP sizes, SHA-256 and member inventories. After restoration, `--require-data` additionally requires every external evidence file and verifies its contents.

## Restore without changing the source

Choose a destination outside the repository and asset directory that does not yet exist:

```sh
.venv/bin/python -B tools/prepare_workspace.py --assets-dir ../github-release-assets --destination /tmp/nsn-work-new --dry-run
```

Remove `--dry-run` to create the new copy and restore the checked ZIP members. The tool refuses to overwrite an existing directory. It performs no numerical analysis or model execution. The source research workspace uses the longer tool path given above, with the same arguments.

## Scientific computation

Use a separate restored copy, its root `.venv`, the fixed specifications and the required dependency versions. `tools/reproduce.py`, the extraction scripts, table and figure builders, and experiment verifiers perform numerical calculations and/or write derived outputs. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Original code retains its historical command examples; this file takes precedence for interpreter selection.

The readable Colab launchers keep the original `sys.executable` and isolated runtime. Do not replace those paths with a Mac `.venv` path. No local model loading is needed for package verification or saved-data analysis.
