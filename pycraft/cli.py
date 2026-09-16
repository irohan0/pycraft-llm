# pycraft/cli.py
#
# Command-line interface for PyCraft-1.
#
#     pycraft generate "def is_palindrome(s):"
#     pycraft fim --prefix "def add(a, b):\n    " --suffix "\n\nadd(1, 2)"
#     pycraft serve --port 8000
#     pycraft chat
#     pycraft info
#
# Equivalent without installing: python -m pycraft.cli <command>

import argparse
import os
import sys


# ------------------------------------------------------------------ #
# Shared arguments
# ------------------------------------------------------------------ #
def _add_model_args(p):
    p.add_argument("--checkpoint", default=None,
                   help="weights directory or .safetensors file "
                        "(default: checkpoints/sft_stage1, else HuggingFace)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"],
                   help="default: cuda when available, else cpu")
    p.add_argument("--quantize", action="store_true",
                   help="dynamic int8 on CPU — roughly 1.4x faster, "
                        "4x smaller, slight quality cost")
    p.add_argument("--threads", type=int, default=None,
                   help="torch intra-op threads (default: torch's own choice)")


def _add_sampling_args(p, max_tokens=200):
    p.add_argument("-n", "--max-tokens", type=int, default=max_tokens)
    p.add_argument("-t", "--temperature", type=float, default=0.2,
                   help="0.0 selects greedy decoding")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)


def _load(args):
    from pycraft.engine import PyCraft
    return PyCraft(
        checkpoint=args.checkpoint,
        device=args.device,
        quantize=args.quantize,
        threads=args.threads,
    )


def _sampling(args) -> dict:
    return dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


# ------------------------------------------------------------------ #
# Commands
# ------------------------------------------------------------------ #
def cmd_generate(args):
    engine = _load(args)
    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()

    if args.raw:
        # Stream straight out so long generations are visible as they arrive
        for delta in engine.stream(prompt, stop=args.stop or None,
                                   **_sampling(args)):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()
    else:
        print(engine.complete_code(prompt, stop=args.stop or None,
                                   **_sampling(args)))
    return 0


def cmd_fim(args):
    engine = _load(args)
    middle = engine.fill_in_middle(
        prefix=args.prefix.replace("\\n", "\n"),
        suffix=args.suffix.replace("\\n", "\n"),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    if args.full:
        print(args.prefix.replace("\\n", "\n") + middle
              + args.suffix.replace("\\n", "\n"))
    else:
        print(middle)
    return 0


def cmd_serve(args):
    # The server reads its configuration from the environment so that a plain
    # `uvicorn pycraft.server:app` behaves exactly like this command.
    if args.checkpoint:
        os.environ["PYCRAFT_CHECKPOINT"] = args.checkpoint
    if args.quantize:
        os.environ["PYCRAFT_QUANTIZE"] = "1"
    if args.threads:
        os.environ["PYCRAFT_THREADS"] = str(args.threads)
    os.environ["PYCRAFT_CONCURRENCY"] = str(args.concurrency)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'pycraft-llm[serve]'",
              file=sys.stderr)
        return 1

    # flush=True: stdout is block-buffered when redirected to a file or pipe,
    # so without it this banner (and the tunnel hint) never appears until the
    # process exits.
    print(f"  PyCraft-1 API on http://{args.host}:{args.port}", flush=True)
    print(f"  interactive docs: http://{args.host}:{args.port}/docs", flush=True)
    if args.host == "127.0.0.1":
        print("  to share temporarily (free, no account):", flush=True)
        print(f"    cloudflared tunnel --url http://127.0.0.1:{args.port}",
              flush=True)
    else:
        print("  WARNING: bound beyond localhost and there is no auth.",
              flush=True)
    uvicorn.run("pycraft.server:app", host=args.host, port=args.port,
                reload=False, log_level=args.log_level)
    return 0


def cmd_chat(args):
    engine = _load(args)
    print("PyCraft-1 interactive. Blank line submits, Ctrl-C or /quit exits.")
    print("PyCraft-1 completes code — it is not a conversational assistant.\n")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip() in ("/quit", "/exit"):
            return 0
        if not line.strip():
            continue
        for delta in engine.stream(line + "\n", **_sampling(args)):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print("\n")


def cmd_info(args):
    engine = _load(args)
    for key, value in engine.info().items():
        print(f"  {key:<16} {value}")
    return 0


# ------------------------------------------------------------------ #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycraft",
        description="PyCraft-1 — a 55M-parameter Python code model "
                    "that runs locally on CPU.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="complete a code prompt")
    p.add_argument("prompt", help="prompt text, or '-' to read stdin")
    p.add_argument("--raw", action="store_true",
                   help="stream unmodified output instead of cleaning it up")
    p.add_argument("--stop", action="append", default=[],
                   help="stop string (repeatable)")
    _add_sampling_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("fim", help="fill in the middle between two snippets")
    p.add_argument("--prefix", required=True, help="code before the gap")
    p.add_argument("--suffix", required=True, help="code after the gap")
    p.add_argument("--full", action="store_true",
                   help="print prefix + middle + suffix, not just the middle")
    _add_sampling_args(p, max_tokens=128)
    _add_model_args(p)
    p.set_defaults(func=cmd_fim)

    p = sub.add_parser("serve", help="run the local REST API")
    p.add_argument("--host", default="127.0.0.1",
                   help="default 127.0.0.1; the API has no authentication")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--concurrency", type=int, default=2,
                   help="max simultaneous generations")
    p.add_argument("--log-level", default="info")
    _add_model_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("chat", help="interactive prompt loop")
    _add_sampling_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("info", help="show the loaded model configuration")
    _add_model_args(p)
    p.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
