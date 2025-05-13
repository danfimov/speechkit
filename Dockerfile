FROM python:3.12.10-slim-bookworm AS build

COPY --from=registry.tochka-tech.com/proxy_ghcr-io/astral-sh/uv:0.6.16 /uv /uvx /bin/

# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups,
# - prevent uv from accidentally downloading isolated Python builds,
# - pick a Python,
# - and finally declare `/project` as the target for `uv sync`.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT="/app/.venv"

USER root
COPY ./pyproject.toml ./uv.lock  ./
RUN uv sync --no-dev --locked

FROM python:3.12.10-slim-bookworm AS production

USER root
RUN apt-get update -y && \
    apt-get install -y libnuma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

COPY --from=build app/.venv /app/.venv/
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000 8001

ENTRYPOINT [ "docker/entrypoint.sh" ]
CMD [ "docker/start.sh" ]
