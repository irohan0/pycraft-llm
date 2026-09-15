# pycraft/server.py
#
# Local REST API for PyCraft-1.
#
#     pycraft serve                       # then http://127.0.0.1:8000/docs
#     uvicorn pycraft.server:app          # equivalent
#
# Binds to 127.0.0.1 by default: this is a local-first service with no
# authentication. To share it temporarily, put a tunnel in front of it
# (see `pycraft serve --help`) rather than binding to 0.0.0.0.
#
# Configuration comes from the environment so that plain `uvicorn
# pycraft.server:app` behaves identically to the CLI:
#
#     PYCRAFT_CHECKPOINT   path to weights (default: checkpoints/sft_stage1)
#     PYCRAFT_QUANTIZE     "1" to enable dynamic int8 on CPU
#     PYCRAFT_THREADS      torch intra-op thread count
#     PYCRAFT_CONCURRENCY  max simultaneous generations (default 2)

import json
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pycraft.engine import PyCraft, strip_fences

_engine: PyCraft | None = None

# Generation is CPU-bound and holds the GIL only intermittently, so unbounded
# concurrency would just thrash the thread pool and make every request slower.
# The KV cache is per-request state, so a semaphore is all the isolation needed.
_slots = threading.Semaphore(int(os.environ.get("PYCRAFT_CONCURRENCY", "2")))


def get_engine() -> PyCraft:
    global _engine
    if _engine is None:
        _engine = PyCraft(
            checkpoint=os.environ.get("PYCRAFT_CHECKPOINT") or None,
            quantize=os.environ.get("PYCRAFT_QUANTIZE") == "1",
            threads=int(os.environ["PYCRAFT_THREADS"])
            if os.environ.get("PYCRAFT_THREADS") else None,
        )
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup so the first request is not the one that pays for it.
    engine = get_engine()
    print(f"  PyCraft-1 ready: {engine.info()}")
    yield


app = FastAPI(
    title="PyCraft-1",
    description="Local inference API for a 55M-parameter Python code model.",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #
class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to continue")
    max_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=2.0,
                               description="0.0 selects greedy decoding")
    top_k: int = Field(20, ge=0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stop: list[str] | None = None
    seed: int | None = None
    strip_markdown: bool = Field(
        True, description="Strip the markdown fences the SFT data taught it")
    stream: bool = False


class FIMRequest(BaseModel):
    prefix: str = Field(..., description="Code before the gap")
    suffix: str = Field(..., description="Code after the gap")
    max_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_k: int = Field(20, ge=0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
def health():
    return {"status": "ok", **get_engine().info()}


@app.get("/v1/models")
def list_models():
    engine = get_engine()
    return {
        "object": "list",
        "data": [{
            "id": "pycraft-1",
            "object": "model",
            "owned_by": "irohan0",
            "context_window": engine.config.max_seq_len,
            "parameters": "55.3M",
            "capabilities": ["completion", "fill-in-the-middle"],
        }],
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    engine = get_engine()
    kwargs = dict(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
        seed=req.seed,
    )

    if req.stream:
        def sse():
            # StreamingResponse runs a sync generator in a threadpool, so the
            # event loop is never blocked by the decode loop.
            with _slots:
                try:
                    for delta in engine.stream(req.prompt, **kwargs):
                        yield f"data: {json.dumps({'text': delta})}\n\n"
                except Exception as exc:                      # noqa: BLE001
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"},
        )

    started = time.time()
    with _slots:
        try:
            text = engine.generate(req.prompt, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if req.strip_markdown:
        text = strip_fences(text)

    return {
        "object": "text_completion",
        "model": "pycraft-1",
        "choices": [{"index": 0, "text": text}],
        "usage": {
            "prompt_tokens": len(engine.tokenizer.encode(req.prompt)),
            "completion_tokens": len(engine.tokenizer.encode(text)),
        },
        "elapsed_s": round(time.time() - started, 3),
    }


@app.post("/v1/fim")
def fill_in_middle(req: FIMRequest):
    """
    Fill in the gap between `prefix` and `suffix`.

    Half of PyCraft-1's pretraining used the Fill-in-the-Middle objective, so
    infilling is a trained capability here rather than a prompting trick.
    """
    engine = get_engine()
    started = time.time()
    with _slots:
        try:
            middle = engine.fill_in_middle(
                prefix=req.prefix,
                suffix=req.suffix,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
        except ValueError as exc:
            # Raised when prefix+suffix cannot fit the 1024-token context
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "object": "fim_completion",
        "model": "pycraft-1",
        "middle": middle,
        "completed": req.prefix + middle + req.suffix,
        "elapsed_s": round(time.time() - started, 3),
    }
