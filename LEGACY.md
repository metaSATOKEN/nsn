# Earlier NSN publication and reference library

The earlier paper is [*Norm Separation is Necessary*, v1.0.0](https://doi.org/10.5281/zenodo.19724817). Its [original README](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/README.md), [paper source](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/paper/nsn.md), [PDF](https://github.com/metaSATOKEN/nsn/blob/dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d/paper/nsn.pdf), and experiment layout remain accessible at the linked commit. The DOI identifies the earlier publication, not the integrated paper.

## Library users

The `src/nsn/` implementation is retained byte for byte from that edition. The library version remains **0.3.12**; **v1.0** identifies the integrated paper/package edition, not a new library API. This repository also contains research evidence and experiment-specific launchers. The library's generic dependencies are separate from the fixed reproduction environments in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For the exact historical library, an installation into your own chosen environment can be pinned to the commit:

```sh
.venv/bin/python -m pip install 'git+https://github.com/metaSATOKEN/nsn.git@dc6b52a0b8698c4d1971273fac309bf8e0cbdd6d'
```
