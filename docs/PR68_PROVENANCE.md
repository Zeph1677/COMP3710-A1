# PR #68 provenance and assessment

Verified on 26 November 2026.

## Original submission snapshot

`original-pr68/` contains all 19 tracked files under `recognition/HipMRI_3D_UNet_s4910970` at PR head commit `6bc9a99a781aa3333cb3d0adb6ed9e36dcb33025` in `Zeph1677/PatternAnalysis-2026`. Files were read from Git objects and copied byte for byte, including the original README, `.gitignore`, `compare_failures.py` and K042 failure image. The source commit identifies the checked PR head; it is not a separate assertion about the marking cutoff.

[Source tree](https://github.com/Zeph1677/PatternAnalysis-2026/tree/6bc9a99a781aa3333cb3d0adb6ed9e36dcb33025/recognition/HipMRI_3D_UNet_s4910970) · [Original PR](https://github.com/shakes76/PatternAnalysis-2026/pull/68) · [Blob manifest](../evidence/pr68-manifest.json).

Existing `src/`, `tests/`, `docs/`, `evidence/` and `Submission-final_v2/` files are retained. The snapshot preserves differences without replacing working versions. The original 22-commit course history remains accessible through the source PR; this update does not recreate those commits in this repository.

## Assessment records

- [First preliminary review](https://github.com/shakes76/PatternAnalysis-2026/pull/68#issuecomment-6906766567): implementation subtotal 18/20.
- [Second preliminary review](https://github.com/shakes76/PatternAnalysis-2026/pull/68#issuecomment-6914820929): implementation subtotal 20/20.
- [Final Marking](https://github.com/shakes76/PatternAnalysis-2026/pull/68#issuecomment-7087681515): 33 marks, 7 deducted, equivalent to 33/40 (82.5%). Posted 22 November 2026 at 23:35 UTC, or 23 November in Brisbane. The comment explicitly says the mark is subject to approval from Shakes. No independent confirmation of an LMS grade is claimed.

The final feedback identifies a missing 2D comparison, missing boundary metrics and limitations in failure analysis, implementation, commenting, documentation, AI disclosure and commit progression. Read the linked comment for the full marking table. This portfolio update does not retroactively satisfy those assessment requirements or change the historical AI disclosure.

## Scope and reproduction

The copied snapshot contains no MRI dataset or trained model weights because those were not tracked in the source directory. Use the official dataset source and the original reproduction commands; data-use conditions still apply. The repository's Apache-2.0 licence is retained at the root. Existing AI usage disclosures are retained without inventing historical prompts or verification records.

The root README links to the snapshot for original results, and separately describes working code and later experiments. Later analyses must not be presented as results available at the original submission deadline.
