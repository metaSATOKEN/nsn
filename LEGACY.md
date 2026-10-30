# Earlier NSN edition and library compatibility

The integrated research package continues the existing `metaSATOKEN/nsn` repository. Its history is retained. The earlier default-branch snapshot is preserved at [`legacy-before-integrated-2026-10-30`](https://github.com/metaSATOKEN/nsn/tree/legacy-before-integrated-2026-10-30) (commit [`dc6b52a`](https://github.com/metaSATOKEN/nsn/commit/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d)). All 15 pre-existing version tags remain unchanged.

The earlier paper is [*Norm Separation is Necessary*, v1.0.0](https://doi.org/10.5281/zenodo.19724817). Its [original README](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/README.md), [paper source](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/paper/nsn.md), [PDF](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/paper/nsn.pdf), and experiment layout remain accessible at that snapshot. The DOI identifies the earlier publication, not the integrated paper.

## Library users

The `src/nsn/` implementation is retained byte for byte from that commit, together with its setuptools installation entrypoint. The library version remains **0.3.12**; **v1.0** identifies the integrated paper/package edition, not a new library API. This repository also contains research evidence and experiment-specific launchers. The library's generic dependencies are separate from the fixed reproduction environments in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

To inspect the earlier distribution, use a fresh destination:

```sh
git clone --branch legacy-before-integrated-2026-10-30 https://github.com/metaSATOKEN/nsn.git nsn-legacy
```

For the exact historical library, an installation into your own chosen environment can be pinned to the commit:

```sh
.venv/bin/python -m pip install 'git+https://github.com/metaSATOKEN/nsn.git@dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d'
```

This command installs the library and its dependencies; it is not a file-only verification command. The repository is private during migration, so access requires the owner's permission. Scientific records and old citation identifiers have not been rewritten.
