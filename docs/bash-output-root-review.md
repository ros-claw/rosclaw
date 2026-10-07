# Bash output review

The custom Bash tool now retains at most 65,536 bytes of valid UTF-8 output at code-point boundaries. Each stream has an incremental decoder that preserves an initial BOM. Once a character cannot fit, later bytes cannot replace it. Excess output is counted and explicitly marked in the model-visible result and structured details. Exit, timeout and cancellation behavior remains covered by actual factory tests.

ROSClaw Native/Kimi authored both product files. Root copied their exact closed bytes, independently replayed saved tool effects and ran 22 actual factory controls, a separate BOM comparison, all 471 compiled Node tests (468 passed, three skipped), and six Python PI bridge tests. The evidence summary is in `reports/bash-output-root-review.json`. No provider call or robot actuation was required for these checks.

The earlier source cutoff, broken empty-literal unit assertion and independently discovered BOM regression remain failed historical records. Their later repairs do not change those results. This validation covers valid UTF-8 and independent stream decoding; it does not certify binary-output fidelity or a total chronology across stdout and stderr.
