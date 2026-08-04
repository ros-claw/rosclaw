# T1 跨 UID operator e2e 镜像（CI 构建；本地无网可跳过）。
FROM python:3.12-slim

WORKDIR /repo
# hatch force-include 引用 e-urdf-zoo/configs/worker_plugins/benchmarks 等
# 多个路径——按 pyproject 全量拷贝（.dockerignore 排除重型产物）。
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY e-urdf-zoo ./e-urdf-zoo
COPY configs ./configs
COPY worker_plugins ./worker_plugins
COPY benchmarks ./benchmarks
RUN pip install --no-cache-dir -e . && \
    rm -rf /root/.cache

ENV ROSCLAW_REPO=/repo
ENTRYPOINT ["python", "scripts/e2e/operator_cross_uid/entrypoint.py"]
