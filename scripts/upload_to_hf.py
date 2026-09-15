# scripts/upload_to_hf.py
#
# Publish the contents of upload_package/ to the HuggingFace model repo.
#
#   hf auth login          # once, interactively — stores the token for you
#   python -m scripts.upload_to_hf --dry-run
#   python -m scripts.upload_to_hf
#
# NEVER hardcode a token in this file. An earlier version did, and the value
# stayed readable in git history long after it was blanked in the working
# copy. Credentials are read only from the HuggingFace CLI login or the
# HF_TOKEN environment variable, and are never printed.

import argparse
import sys
from pathlib import Path

UPLOAD_DIR = Path("upload_package")
DEFAULT_REPO = "imshadow0/pycraft-1"

# Everything else in upload_package/ is uploaded as-is.
SKIP = {".git", "__pycache__", ".ipynb_checkpoints"}


def collect_files() -> list[Path]:
    files = []
    for p in sorted(UPLOAD_DIR.rglob("*")):
        if p.is_file() and not any(part in SKIP for part in p.parts):
            files.append(p)
    return files


def resolve_token() -> str | None:
    """
    Token from the HF CLI login, or HF_TOKEN. Returns None if neither is set,
    so the caller can print instructions rather than a stack trace.
    """
    import os
    from huggingface_hub import get_token
    return os.environ.get("HF_TOKEN") or get_token()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload upload_package/ to the HuggingFace model repo")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    ap.add_argument("--message", default="Fix CPU inference; add KV cache, "
                                         "FIM example, HumanEval results")
    args = ap.parse_args()

    if not UPLOAD_DIR.is_dir():
        print(f"ERROR: {UPLOAD_DIR}/ not found. Run from the repo root.",
              file=sys.stderr)
        return 1

    files = collect_files()
    if not files:
        print(f"ERROR: {UPLOAD_DIR}/ is empty.", file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in files)
    print(f"  repo   : {args.repo}")
    print(f"  files  : {len(files)}  ({total/1e6:.1f} MB)")
    for f in files:
        print(f"    {f.relative_to(UPLOAD_DIR).as_posix():<40} "
              f"{f.stat().st_size/1e6:8.2f} MB")

    if args.dry_run:
        print("\n  dry run — nothing uploaded.")
        return 0

    token = resolve_token()
    if not token:
        print("\nERROR: no HuggingFace credentials found.\n"
              "  Run:  hf auth login\n"
              "  (or set HF_TOKEN in the environment)\n"
              "Create a fine-grained token with Write access scoped to just\n"
              f"{args.repo} at https://huggingface.co/settings/tokens",
              file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    who = api.whoami()
    print(f"\n  authenticated as: {who.get('name', '?')}")

    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    print(f"  uploading to {args.repo} ...")
    api.upload_folder(
        folder_path=str(UPLOAD_DIR),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
        ignore_patterns=[f"**/{s}/**" for s in SKIP],
    )
    print(f"  done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
