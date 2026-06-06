FROM pytorch/pytorch:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && uv pip install --system --no-cache git+https://github.com/freud14/poutyne.git@stable \
    && find /opt/conda/lib/ \( \
        -name '*.a' -o -name '*.pyc' -o -name '*.c' -o -name '*.pxd' \
        -o -name '*.pyd' -o -name '*.js.map' -o -name '*.mc' \
        -o -name '*.md' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \
    \) -delete \
    && find /opt/conda/lib/ -name '__pycache__' -exec rm -rf {} +
