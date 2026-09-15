# Running PyCraft-1 locally

PyCraft-1 is 55.3M parameters and runs on CPU. There is no hosted service and
nothing here costs money to run.

## Install

```bash
git clone https://github.com/irohan0/pycraft-llm
cd pycraft-llm
pip install -e ".[serve]"
```

The editable install is the supported path: `model` and `tokenizer` are
generic top-level package names, kept where they are so `training/` and
`eval/` continue to work unchanged.

Weights resolve from `checkpoints/sft_stage1` when you run from a clone, and
fall back to downloading from HuggingFace otherwise (needs `pip install -e ".[hub]"`).

## Command line

```bash
pycraft info
pycraft generate "# Task: reverse a list\n\ndef reverse_list(xs):\n"
pycraft fim --prefix 'def square(n):\n    ' --suffix '\n\nprint(square(4))' --full
pycraft chat
pycraft serve --port 8000
```

Useful flags: `--quantize` (int8, ~1.4x faster on CPU), `--threads N`,
`-t/--temperature` (0.0 is greedy), `-n/--max-tokens`, `--stop` (repeatable).

## REST API

```bash
pycraft serve                  # http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/completions` | Continue a prompt, or a list of prompts as one batch. Set `"stream": true` for SSE (single prompt only). |
| `POST /v1/fim` | Fill the gap between `prefix` and `suffix`. |
| `GET /health` | Liveness plus the loaded configuration. |
| `GET /v1/models` | Model metadata. |

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"# Task: add two numbers\n\ndef add(a, b):\n",
       "max_tokens":60,"temperature":0.0,"stop":["\n\n"]}'
```

**Fill-in-the-Middle** is the endpoint worth knowing about. Half of
PyCraft-1's pretraining used the FIM objective, so infilling is trained
behaviour rather than a prompting trick:

```bash
curl -X POST http://127.0.0.1:8000/v1/fim \
  -H 'Content-Type: application/json' \
  -d '{"prefix":"def factorial(n):\n    if n <= 1:\n        return 1\n    ",
       "suffix":"\n\nprint(factorial(5))\n"}'
# -> {"middle": "return n * factorial(n-1)\n\n", ...}
```

The server binds to `127.0.0.1` and has **no authentication**. To share it
temporarily, put a free tunnel in front of it rather than binding wider:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

## Docker

```bash
docker build -t pycraft .
docker run --rm -p 127.0.0.1:8000:8000 pycraft
```

CPU-only image. Weights are baked in; mount `checkpoints/` instead to keep the
image small.

## Performance (measured, RTX 3050 laptop running CPU-only, 8 threads)

Generation uses a KV cache, so per-token cost is flat rather than growing with
context:

| Context | Without cache | With cache |
|---|---|---|
| 32 | 31.0 tok/s | 61.9 tok/s |
| 256 | 12.3 tok/s | 53.7 tok/s |
| 512 | 6.6 tok/s | 48.6 tok/s |

On a realistic workload (200-token prompt, 400 generated) this is **8.8x**:
48.5s becomes 5.5s. `--quantize` adds roughly 1.4x on top (68 -> 94 tok/s).

Batching several prompts multiplies throughput again, because per-token cost
is dominated by weight loading that every sequence in the batch shares:

| Batch size | Aggregate |
|---|---|
| 1 | 65.7 tok/s |
| 2 | 129.9 tok/s |
| 4 | 186.7 tok/s |
| 8 | 267.7 tok/s |

Pass a list to `/v1/completions` (or `PyCraft.generate_batch`) to use it.
Prompts are left-padded internally and each result is cut at its own EOS, so
mixed lengths are fine:

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":["def add(a, b):\n","def rev(s):\n"],"max_tokens":40}'
# -> {"choices":[{"index":0,...},{"index":1,...}], ...}
```

## Limitations

- 1024-token context, and it is a **hard stop**, not a sliding window. The KV
  cache stores post-RoPE keys, which cannot be re-based without re-rotating
  every cached key.
- 55M parameters: good on standard algorithms and common library patterns,
  unreliable on anything longer. It completes code; it is not a chat assistant.
- The SFT data (Magicoder) was full of markdown fences, so raw output often
  contains them. The CLI and API strip them by default.
