# W11（一）打包断点修复：共同 staging 与 wheel 自包含（规格 §15.1/§15.4）

**分支**：a-w11-packaging（基于 main=8221d27，§17：W11 从 W00 起并行准备）
**日期**：2026-09-09

## 断点（修复前实证）

1. `scripts/build_release.sh` 在**开发 checkout** 里 `npm ci`
   （两次：build + prod pack）——构建后开发树 node_modules 被删/
   改写（规格 §15.1 明令禁止）。
2. PyPI wheel 的 force-include **不含 packages/**——`pip install
   rosclaw` 拿不到 agent JS，无法启动 Native Agent（规格 §15.1
   「不能被假定自动包含」实证）。
3. sdist 缺 packages//scripts/js-stage——无法从自身内容重建
   wheel（规格 §15.4）。

## 改动

- **`scripts/release/build_js_staging.sh`（新）**：共同 JS
  staging——rsync 副本进专用工作目录（`dist/.js-build`），在其内
  npm ci + clean build + prod pack；产出 dist/package.json/
  lock/src/node_modules.prod.tar.gz + 完整性硬校验。
  `build_release.sh` 改为消费 staging（开发 checkout 全程不被
  npm ci 触碰）。copy-assets 的 python-tree 相对路径在 staging
  不成立——喂包内 prompts/ fallback（该 mjs 的设计路径）。
- **`pyproject.toml`**：wheel force-include `dist/js-stage` 的
  两包 dist + package.json + lock（缺 staging = 构建硬错误）；
  node_modules 不进 wheel；sdist include += /packages /scripts
  /third_party + force-include `dist/js-stage`（gitignore 产物
  必须 force-include 才进 sdist）。
- **`src/rosclaw/agentd/pi_entry.py`**：dist 入口解析加第 4 来源
  ——wheel 内嵌 `rosclaw/js_stage/<pkg>/dist/src/main.js`
  （env → 仓库 → 安装前缀 → wheel 内嵌）。
- **CI Build Package**：Set up Node 22.19 + 先跑 JS staging 再
  `hatch build`；wheel 校验加 js_stage 必需项 + 无 node_modules；
  新增 sdist 自包含校验（重建输入实际开包检查）。
- **README**：Alpha 安装形式如实说明（tar 自包含；PyPI wheel
  需 Node ≥22.19 前置、无聊天期下载、GL 系统依赖见 doctor——
  不宣传「pip 安装后零系统依赖」）。

## 验证（红→绿 + 实开包）

| 验证 | 结果 |
|---|---|
| tests/test_w11_packaging.py（6：无开发 checkout npm ci/staging 专用工作目录/wheel force-include JS/sdist 自包含/wheel 内嵌解析/解析顺序） | 红（6 failed）→ 绿 6/6 |
| 真实 staging（npm ci×2 + build×2 + prod pack×2） | 绿；开发 checkout 不被触碰（工作目录在 dist/.js-build） |
| 实开 wheel（pip wheel 后 zipfile 检查） | 1958 文件；js_stage 293 文件；两包 main.js/prompts/lock 在场；无 node_modules；tag=py3-none-any（Alpha 无平台二进制——正确） |
| 干净 venv 安装 wheel → 内嵌入口解析 | agent/tui 均 True |
| sdist 实开包 | 3634 条目；重建输入全在场 |
| **sdist→wheel 重建**（unpack + pip wheel --no-build-isolation） | 重建 wheel 含 js_stage main.js |
| ruff check src tests | 全过 |

## 未跑/后续（W11 剩余）

- BuildManifest 动态化核查、Node/fd/rg 下载校验、体积测量 vs
  PyPI 100MB、x86_64 认证矩阵（CI Build Package 在 x86_64 跑
  staging+wheel 校验——本 PR 起生效）、自包含 wheel（bundled
  Node runtime per-platform）评估。
- 本机 /tmp/node2219 的 npm 残缺（7/71 commands——npm config
  MODULE_NOT_FOUND）；本地构建改用 /home/nvidia/.local/node
  （v24.19.0 npm 11.17.0 完整）。CI 用 setup-node 不受影响。
- 真实模型 journey：**NOT_RUN**（无 key，不合成冒充）。
