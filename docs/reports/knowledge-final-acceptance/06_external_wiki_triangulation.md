# External Wiki triangulation

Status: PASS for coverage/cross-checking; derived sources remain non-canonical.

The same focused question was sent to DeepWiki and GitMCP for three projects,
while direct GitHub snapshots supplied the authoritative evidence.

| Project/question focus | DeepWiki | GitMCP | Direct source |
|---|---|---|---|
| unitree_rl_lab: G1 training and sim-to-sim deployment | G1/train/deploy present, 3 docs, 515,120 bytes | all terms present, 1 focused doc | pinned commit, 10/10 claim audit |
| CodeWiki: hierarchy and incremental generation | hierarchy/increment/generate present, 3 docs, 433,133 bytes | all terms present, 3 docs | pinned commit, 10/10 claim audit |
| GraphRAG: local/global/DRIFT modes | local/global/DRIFT present, 3 docs, 849,966 bytes | all terms present, 3 docs | pinned commit, 10/10 claim audit |

All six derived snapshots reported authority B and `code_executed=false`.
DeepWiki was stronger for broad repository structure; GitMCP often returned a
smaller focused result; ROSClaw Know retained the strongest version/provenance
story because every accepted fact resolved to the pinned direct snapshot.

No automatic corroboration bonus changes truth. A conflicting derived answer
must enter the disagreement queue and be verified against the direct source.
