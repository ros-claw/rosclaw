# G5 v1 negative findings — original-record forensics

The original 54-episode study is an honest negative/unsubstantiated result. A succeeded in 0/18 tasks, B in 1/18, C in 0/18. This audit changes no original verdict and calls no model or World. A is the same requested model with restricted external generic tools; it is not the complete Codex product.

All 546 saved decisions were joined one-to-one with their original CLI events, usage, prompt and tool trace. Each saved transcript was independently reconstructed. Every one of the 30 byte-budget exits has a replayed next prompt exceeding 48,000 UTF-8 bytes; all 23 tool-budget exits have exactly 12 turns. These are outer harness limits, not evidence that the model's native context window was exhausted.

| Operation | Original attempts |
|---|---:|
| list_files | 54 |
| read_file | 416 |
| query_memory | 23 |
| ros_inspect_system | 11 |
| ros_diagnose | 17 |
| ros_context | 5 |
| select_interfaces | 9 |
| propose_patch | 10 |
| run_tests | 1 |
| finish | 0 |

List/read accounts for 470/546 = 86.08% of turns. The remaining 76 turns include Memory, diagnostics and invalid operations; they are not all productive engineering decisions. There were 44 rejected tool requests: 29 invalid file slices, five missing Memory query arguments, two invalid patch lists, and eight nonexistent requested paths/keys. There were 36 oversized results sent through the file indirection protocol. Largest measured expert output: 86,620 UTF-8 bytes. The audit finds 28 overlapping read turns and 65,590 repeated characters under the same file+JSON-pointer coordinate system; overlaps between different pointer representations are not counted.

Eleven episodes produced a statically accepted patch/interface proposal, first occurring at median turn 10 (range 8–12). Static acceptance is not a correct diagnosis, necessary change or successful repair. Only one episode requested a new physical validation and obtained canonical success. The other 53 are upstream harness/task failures, not 53 physical navigation failures.

Memory had 23 attempts, 18 successful queries, and 15 episodes with admitted records. No model decision explicitly referenced an admitted Memory alias. This does not prove the model ignored the text or that Memory could not affect a decision implicitly; there is no supported causal benefit claim.

| Arm | Median initial capture | Median model+tools phase | Median total wall | Static proposal episodes | New validation episodes |
|---|---:|---:|---:|---:|---:|
| A | 17.42 s | 142.21 s | 159.60 s | 3/18 | 0/18 |
| B | 17.39 s | 110.75 s | 129.48 s | 4/18 | 1/18 |
| C | 17.42 s | 136.87 s | 152.38 s | 4/18 | 0/18 |

The median total wall across all episodes is 148.80 seconds. No episode exits due to the wall-time budget. Increasing that budget alone would not address the observed byte/tool exits. Separate per-call model, tool and physical-validation durations were not fully instrumented; enclosing phase clocks are measured, the isolated durations remain UNKNOWN. Filesystem mtime is not substituted for missing clocks. Medians of separate phases need not add to the median total.

The evidence supports an interface bottleneck hypothesis: large raw-file results, repeated transcript serialization, unclear argument contracts and mandatory Memory interaction consumed decisions before verification. It does not establish the counterfactual success rate of a better interface. No model, navigation-controller or daemon changes are justified solely by these negative results.

Next work is a separate, pure read-only engineering-case compiler and replay/control tests. Typed repair/ACI integration and a new pilot require separate preregistration; no new result will replace these 54 originals.

Reproduce with `PYTHONPATH=src:. .venv/bin/python -m benchmarks.ros_expert_ablation.forensics.audit INPUT OUTSIDE_INPUT_OUTPUT`. Full per-arm/case/seed/turn data, input hashes, evidence refs and missingness are in the adjacent JSON files. Public original artifacts are sealed separately with a manifest and SHA256; operator credentials, SDK private keys, ledger and databases are excluded.
