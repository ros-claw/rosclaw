# ROSClaw Hub Phase 6 帮助清单（HELP.md）

> 生成时间：2026-06-22  
> 范围：`src/rosclaw/hub/`、`tests/hub/`、`docs/hub/`、`reports/hub_*.md`

## 1. 已完成的工作

- 对照 `/home/ubuntu/.claude/plans/stateless-discovering-quill.md` 实施指南，完成 Phase 1–6 全部交付物：
  - Schema / refs / CLI / cache / index / fake registry / auth / verifier / permissions / licenses / lifecycle / lockfile / resolver / installer / registry_writer / mcp_merge / health / publisher。
- 补齐了原先低覆盖模块的测试：`cli.py`、`health.py`、`lifecycle.py`、`auth.py`、`verifier.py`、`registry_writer.py`、`client.py`、`publisher.py` 均已有对应测试文件。
- 修复了 Phase 6 过程中发现的问题：
  - `installer.py:_rollback()` 现在会删除并保存 lockfile 条目，避免部分安装状态残留。
  - `tests/hub/test_client.py` 修正了 `MagicMock` 上下文管理器返回值。
  - `tests/hub/test_publisher.py` 修正了 `bundle()` 输出目录需预先创建的问题。
  - `cache.py` / `lockfile.py` 对 `resolve_home()` 结果做 `cast(Path, ...)`，使 `mypy src/rosclaw/hub` 通过。
  - **`tests/fixtures/fake_registry/server.py` 现在会完整摄入 HTTP 上传的 `.rosclaw` bundle**：解压 manifest、写入 `manifests/...`、复制 bundle 到 `bundles/...`、内容寻址写入 blobs、追加 `catalog.jsonl`，使 publish → sync → install 的 HTTP 端到端闭环可用。
- 在 `pyproject.toml` 中增加了 Hub 范围的 mypy 配置，实现 `mypy src/rosclaw/hub` clean。
- 更新/补充了文档与报告：
  - `docs/hub/*.md`
  - `reports/hub_progress.md`
  - `reports/hub_validation_report.md`
  - `README.md` / `QUICKSTART.md` 增加 Hub Quickstart
  - `.github/workflows/ci.yml` 增加 `hub-test` job
- **PR #18 已合并到 `main`**（merge commit `2fcec56e`，2026-06-21T19:54:52Z）。

## 2. 当前状态

- **代码状态**：Hub 子系统功能完整，CI 中 `hub-test` job 运行 `pytest tests/hub -v`。
- **测试结果**：`pytest tests/hub -q` → **287 passed**。
- **类型检查**：`mypy src/rosclaw/hub` → clean。
- **Lint / Format（Hub 范围）**：`ruff check src/rosclaw/hub tests/hub tests/fixtures/fake_registry/server.py` 与 `ruff format --check src/rosclaw/hub tests/hub tests/fixtures/fake_registry/server.py` 均通过。
- **PR 状态**：PR #18 已合并；报告已同步为 MERGED 状态。
- **环境限制**：当前机器无法新建完整 dev venv（`rosclaw-how` 拉取 torch/cuda 导致 `/` 磁盘满），因此使用 `PYTHONPATH=src:.` 配合系统已装 `pytest`/`ruff`/`mypy` 验证。

## 3. 对照实施指南仍要开发的 / 未关闭的 gap

| # | 事项 | 说明 / 优先级 |
|---|------|--------------|
| 1 | **真实云端 Registry 客户端** | 目前只有 `FakeRegistryClient`；CloudRegistryClient 仅是接口/占位。实施指南明确 deferred，但需要后端 API 规范才能继续。 |
| 2 | **生产级签名替换** | 当前使用 `DUMMY_SIGNING_KEY` / `DUMMY_CERT_PEM`，必须替换为 Sigstore / cosign 或 HSM。 |
| 3 | **大模型 weight 断点续传** | 实施指南 deferred 项：需要设计 Range 请求 / resume / 存储方案。 |
| 4 | **body.yaml patch 与 BodyResolver 深度集成** | 目前写入 patch 后需手动应用，需与 `BodyResolver` / RuntimeClient 自动对接。 |
| 5 | **全仓库格式化** | `ruff format --check .` 仍因历史文件失败，需要决定是否全仓库格式化或维持仅 Hub scope。 |
| 6 | **开发依赖安装** | `pip install -e ".[dev]"` 因 `rosclaw-how` 导致 OSError 磁盘满，需要调整依赖/CI 缓存/分区。 |
| 7 | ~~在最新 main 上完整跑 Phase 6 acceptance 命令~~ | ✅ 已完成：已修复 fake registry HTTP 上传闭环并跑通 publish → sync → search → install → list → uninstall。 |

## 4. 需要什么 / 需要谁支持

- **Cloud Registry 团队**：提供 registry API 规范、认证方式（token / OIDC）、manifest/bundle/catalog 的 endpoint 设计。
- **安全/基础设施团队**：确定生产签名方案（Sigstore/cosign/HSM）及密钥管理流程；移除/替换 `DUMMY_SIGNING_KEY`。
- **模型/存储团队**：确定大模型 weight 的存储、分发、断点续传策略（对象存储、CDN、Range / resume）。
- **RuntimeClient/BodyResolver 负责人**：对接 body.yaml patch 的自动化应用流程。
- **CI/环境支持**：解决 dev venv 磁盘不足问题（如 CI 镜像预装 torch、启用依赖缓存、或调整 `rosclaw-how` 安装范围）。

## 5. 已解决 / 不再困惑的问题

- ✅ PR #18 状态已确认：已合并到 `main`。
- ✅ `tarfile.extractall()` 弃用警告已通过 `src/rosclaw/hub/_compat.py` 中的 `extractall_tar(filter="data")` 兼容性辅助函数解决。
- ✅ `client.py` / `publisher.py` 的测试覆盖缺口已被 `test_client.py` / `test_publisher.py` 补齐。
- ✅ **fake registry HTTP 上传闭环已修复**：`tests/fixtures/fake_registry/server.py` 现在会摄入 `.rosclaw` bundle 并更新 catalog，Phase 6 acceptance 命令已在最新 `main` 上跑通。

## 6. 困惑 / 风险

- **全仓库 lint/format 范围**：仅 Hub scoped 通过不等于整体仓库通过，CI 是否足够？需要团队决策。
- **占位签名材料**：`DUMMY_SIGNING_KEY` 仍留在源码，存在被误用于非测试环境的风险。
- **开发环境磁盘**：/data 已迁移 Docker data-root，但 `/` 仍可能因 torch/cuda 安装爆满，影响本地 dev venv 创建。

## 7. 建议的下一步

1. 挑选一个 deferred 项开始设计（真实 cloud registry 或生产签名），等待团队输入。
2. 决定全仓库 `ruff format` 策略。
3. 解决 dev venv 磁盘不足问题，使 `pip install -e ".[dev]"` 可在本地完整运行。
