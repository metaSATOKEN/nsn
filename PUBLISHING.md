# Distribution status and publication checklist

The integrated v1.0 edition is being migrated to [https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn). The repository is private during migration. Existing history and 15 version tags are retained; [LEGACY.md](LEGACY.md) identifies the pre-migration snapshot and old library installation path.

## Remaining distribution steps

1. Attach all 17 ZIPs in `metadata/release-assets.json`, plus `SHA256SUMS`, to a separate integrated-edition Release. The companion assets total 6.30 GB and stay outside Git history. At this stage, attachment is pending.
2. Record the resulting Release URL and asset URLs, then reseal `metadata/package-files.json` and verify the repository and asset inventory. Do not change raw scientific records or frozen generator inputs.
3. Change repository visibility only when the author is ready for public access. Neither the private upload nor a private Release makes the evidence publicly accessible.
4. Optionally archive the integrated edition with Zenodo. The earlier DOI `10.5281/zenodo.19724817` identifies the earlier paper and must not be reassigned to this integrated edition.

Current Markdown papers, figures, code, evidence indexes, and readable experiment launchers are included. Code uses Apache-2.0; author-owned papers, figures and data use CC BY 4.0; third-party terms are retained. Funding and competing-interest declarations were confirmed by the author on 29 October 2026.

## PDFs

The current research-archive PDFs still contain an obsolete contact address and are excluded. Re-export from the current Markdown with **info@metaclan.jp** before including PDFs. Markdown distribution does not depend on that separate export step.

## Verification boundary

The migration preserves experimental code, results, figures, fixed plans and historical provenance. Verification checks file identity and package structure, without new model calls or scientific reanalysis. See [verification and reproduction](REPRODUCIBILITY.md).
