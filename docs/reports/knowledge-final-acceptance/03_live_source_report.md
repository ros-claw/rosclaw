# Live source report

Status: PASS.

The unified opt-in suite (`ROSCLAW_RUN_LIVE_KNOWLEDGE=1`) completed 7/7 in
106.52 seconds:

- pinned GitHub repository snapshot;
- versioned arXiv paper snapshot;
- DeepWiki Public MCP;
- GitMCP;
- Context7;
- two native SeekDB server cases.

DeepWiki, GitMCP, and Context7 each returned derived documents marked Tier B,
`code_executed=false`, and
`evidence_policy=derived_requires_pinned_primary_source`. Their output can aid
discovery and explanation but cannot outrank pinned official evidence as truth.

The primary paper was RoboNaldo, pinned as arXiv `2606.11092v3`. Its stored
document hash was
`cbee4af7c9e3e1c9053790799c1559eba3bd75c52bb10e5da620228d4834d063`.
Motion guidance, curriculum learning, Unitree G1, and football/kicking terms
were all present.

The public MCP endpoints were exercised over Streamable HTTP with exact
read-only tool allowlists. No authentication secret is present in any saved
artifact.
