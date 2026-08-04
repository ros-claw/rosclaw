# T1 跨 UID operator e2e 镜像（CI 构建；本地无网可跳过）。
FROM python:3.12-slim

WORKDIR /repo
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . && \
    rm -rf /root/.cache
COPY scripts ./scripts

ENV ROSCLAW_REPO=/repo
ENTRYPOINT ["python", "scripts/e2e/operator_cross_uid/entrypoint.py"]
