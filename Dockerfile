# ROSClaw v1.0 - Production Docker Image
# Universal OS for Software-Defined Embodied AI

FROM python:3.12-slim as base

LABEL maintainer="ROSClaw Team <team@rosclaw.io>"
LABEL version="1.0.0"
LABEL description="ROSClaw - Universal OS for Software-Defined Embodied AI"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rosclaw

# Install Python dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY bin/ ./bin/

# Re-install after source code is present
RUN pip install --no-cache-dir -e ".[dev]"

# Create runtime directories
RUN mkdir -p /opt/rosclaw/practice_data /opt/rosclaw/rosclaw_data /opt/rosclaw/models

# Non-root user for security
RUN groupadd -r rosclaw && useradd -r -g rosclaw rosclaw \
    && chown -R rosclaw:rosclaw /opt/rosclaw
USER rosclaw

# Default environment
ENV PYTHONPATH=/opt/rosclaw/src
ENV ROSCLAW_WORKDIR=/opt/rosclaw
ENV ROSCLAW_PRACTICE_DIR=/opt/rosclaw/practice_data

EXPOSE 8080

ENTRYPOINT ["rosclaw"]
CMD ["run"]
