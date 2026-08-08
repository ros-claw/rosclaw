# Self-bootstrap: DeepWiki adapter

Status: PASS.

The implementation was checked against the official DeepWiki MCP document and
then exercised against the public server. The final design uses Streamable
HTTP `/mcp` and allowlists exactly:

- `read_wiki_structure`;
- `read_wiki_contents`;
- `ask_question`.

The resulting documents are untrusted Tier B data and require pinned direct
source verification. The adapter performs no repository code execution and
exposes no write tool.

An actual Know/How self-bootstrap replay produced:

- Reference Pack `reference_pack_4084d0de0a89ecfef308211e`;
- opened evidence `self-ev-0`;
- pinned source version `document_version:2026-08-06`;
- How advice `advice_4ede75c8038fe41d10ef60c0`;
- one cited recommendation, no abstention, no private reasoning.

The live DeepWiki test and the three-project triangulation passed. The code
change is in rosclaw-know merge `3e15dfb`.
