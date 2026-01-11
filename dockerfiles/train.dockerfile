FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Use the local cache for uv packages
# instead of downloading them every time
ENV UV_LINK_MODE=copy

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY data data/

RUN --mount=type=cache,target=/root/.cache/uv uv sync


ENTRYPOINT ["uv", "run", "src/diabetic_classification/train.py"]
