FROM pytorch/pytorch:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && uv pip install --system --no-cache git+https://github.com/freud14/poutyne.git@stable \
    && apt-get autoremove -y git
