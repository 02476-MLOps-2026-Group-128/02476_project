FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.diabetic_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
