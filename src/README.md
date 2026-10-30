# NSN reference implementation

Start with [quantize.py](nsn/quantize.py), then [cache.py](nsn/cache.py) and [pack.py](nsn/pack.py). These are the unchanged reference modules used in the earlier configuration and real-storage experiments. Their package-relative imports remain intact.

The small [paper QDQ core](../papers/nsn-integrated/code/nsn_int3_core.py) describes the later packed-model reference. The experiment-specific implementations retain their own arithmetic and storage contracts; the integrated paper explains these differences.

For reproducing reported numbers, use the [saved-data reproduction guide](../REPRODUCIBILITY.md). These modules are source evidence, not a promise of a drop-in cache replacement for every model.
