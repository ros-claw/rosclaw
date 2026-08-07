# Project Wiki audit

Status: PASS.

Each project was pinned to a 40-character commit and audited for ten critical
claim categories. The final campaign processed 26 bounded documents per
repository and passed 70/70 claims.

| Repository | Commit | Claims |
|---|---|---:|
| ros-claw/rosclaw | `0e0a6bd8c022fa875d04bab1250d88c53b5d041c` | 10/10 |
| unitreerobotics/unitree_rl_lab | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` | 10/10 |
| realsenseai/realsense-ros | `60c850958d651130fc2cc3d10efb37ff5be93da5` | 10/10 |
| FSoft-AI4Code/CodeWiki | `d94cacf678792ca4402311d4cb57ba8e3cb9b61a` | 10/10 |
| microsoft/graphrag | `14a00ad88fc33cf2b52f4f113f25807556f8e25e` | 10/10 |
| upstash/context7 | `8d52608e4e27557e6c1e807c8241cffb5544a9a3` | 10/10 |
| stanford-oval/storm | `fb951af7744dab086e34962e9bc6fe878e145f83` | 10/10 |

Gate result: invented files 0; invented symbols 0; missing EvidenceRef 0;
snapshot/hash error 0; severe version error 0; severe fact misread 0.

During audit, `indexed_component_count` was found to cite the first indexed
file rather than a directly supporting fact inventory. The compiler—not the
generated Wiki—was fixed. It now materializes a deterministic, snapshot-bound
`.rosclaw/repo_facts.json` containing files, symbols, imports, entrypoints,
versions, config keys, release rows, component paths, and source hashes.
The entire 70-claim campaign was then rerun successfully.

The resulting minimal regression manifest is stored in rosclaw-know at
`tests/fixtures/real_snapshots/final_acceptance_manifest.json`; it contains
commits, hashes, selected fact excerpts, and expected counts without copying
third-party repositories.
