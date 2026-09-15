# PyCraft-1 local inference API — CPU only.
#
# A 55M-parameter model has no business pulling a multi-gigabyte CUDA image,
# so this installs the CPU-only torch wheel from PyTorch's CPU index.
#
#   docker build -t pycraft .
#   docker run --rm -p 8000:8000 pycraft
#   curl http://127.0.0.1:8000/health
#
# Weights are baked in from checkpoints/sft_stage1 (221 MB). To keep the image
# small instead, drop that COPY and mount them at runtime:
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints" pycraft

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first, as its own layer — it is by far the largest dependency
# and rarely changes, so this keeps rebuilds fast.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.1

COPY pyproject.toml README.md ./
COPY pycraft/     ./pycraft/
COPY model/       ./model/
COPY tokenizer/   ./tokenizer/
COPY data/        ./data/

RUN pip install ".[serve]"

# Weights and vocabulary
COPY tokenizer/vocab/tokenizer.json        ./tokenizer/vocab/tokenizer.json
COPY checkpoints/sft_stage1/               ./checkpoints/sft_stage1/

EXPOSE 8000

# 0.0.0.0 is required inside a container to be reachable from the host; the
# published port is what actually controls exposure. There is no auth, so
# publish to 127.0.0.1 (-p 127.0.0.1:8000:8000) unless you intend otherwise.
ENV PYCRAFT_CONCURRENCY=2

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "pycraft.server:app", "--host", "0.0.0.0", "--port", "8000"]
