# T1 跨 UID operator e2e 镜像（CI 构建；本地无网可跳过）。
FROM python:3.12-slim

WORKDIR /repo
# hatch 的 forced-include 引用 worker_plugins 等路径——必须全量拷贝。
COPY pyproject.toml README.md ./
COPY src ./src
COPY worker_plugins ./worker_plugins
COPY scripts ./scripts
RUN pip install --no-cache-dir -e . && \
    rm -rf /root/.cache

ENV ROSCLAW_REPO=/repo
ENTRYPOINT ["python", "scripts/e2e/operator_cross_uid/entrypoint.py"]
