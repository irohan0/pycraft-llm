# eval/humaneval_runner.py — final clean version

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from model.config import get_config_120m
from model.pycraft_model import PyCraftModel
from tokenizer.tokenizer_utils import PyCraftTokenizer

SFT_CHECKPOINT = "checkpoints/sft_stage1"
RESULTS_DIR = Path("eval/humaneval_results")
MAX_NEW_TOKENS = 400    # generous — prevents truncation
TEMPERATURE = 0.1
TOP_K = 10


def load_model(device):
    tokenizer = PyCraftTokenizer()
    cfg = get_config_120m()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.dropout = 0.0
    model = PyCraftModel(cfg).to(device)
    model.load_state_dict(
        load_file(f"{SFT_CHECKPOINT}/model.safetensors", device=device)
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_body(model, tokenizer, prompt: str, device: str) -> str:
    ids = tokenizer.encode(prompt)
    max_p = 1024 - MAX_NEW_TOKENS
    ids = ids[-max_p:] if len(ids) > max_p else ids
    inp = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    out = model.generate(
        inp, max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE, top_k=TOP_K,
    )
    new_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def extract_body_lines(raw: str) -> list:
    """Extract clean indented body lines from raw generation."""
    raw = re.sub(r'```[\w]*\s*', '', raw)
    raw = re.sub(r'```', '', raw)

    lines = raw.split('\n')
    body = []
    blank_count = 0
    found_code = False

    for line in lines:
        stripped = line.strip()

        if not found_code:
            if stripped == '':
                continue
            if (stripped.startswith(('from ', 'import ', 'def ',
                                     'class ', '#', '"""', "'''"))):
                continue
            if line.startswith(('    ', '\t')):
                found_code = True
            else:
                continue

        if not found_code:
            continue

        if stripped == '':
            blank_count += 1
            if blank_count >= 2:
                break
            body.append('')
        elif line.startswith(('    ', '\t')):
            blank_count = 0
            body.append(line)
        elif stripped.startswith(('def ', 'class ')):
            break
        else:
            break

    # Remove trailing blanks
    while body and body[-1].strip() == '':
        body.pop()

    # Detect truncation — last line ends mid-expression
    if body:
        last = body[-1].strip()
        truncated = (
            not last or
            last.endswith((',', '+', '-', '*', '/', '(', '[', '{',
                           'and', 'or', 'not', 'in', ':')) or
            (last.endswith(('num', 'val', 'ele', 'item', 'char'))
             and 'return' in last)
        )
        if truncated:
            body = ['    pass  # model output truncated']

    return body if body else ['    pass']


def get_clean_prompt_header(he_prompt: str) -> str:
    """
    Extract ONLY the function signature line from he_prompt.
    Crucially: do NOT include any stub body that HumanEval
    prompts sometimes contain (e.g. HumanEval/10 has a partial body).

    Returns: imports + blank line + def line + colon only.
    """
    import_lines = []
    sig_line = ''

    for line in he_prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('from ', 'import ')):
            import_lines.append(stripped)
        if stripped.startswith('def ') and not line.startswith(' '):
            # Take only the def line, not the body
            sig_line = line.rstrip()
            if not sig_line.endswith(':'):
                sig_line += ':'
            break

    parts = []
    if import_lines:
        parts.append('\n'.join(import_lines))
    if sig_line:
        parts.append(sig_line)
    return '\n'.join(parts)


def get_docstring(he_prompt: str) -> str:
    """Extract the docstring from the original prompt."""
    doc_match = re.search(r'(    """.*?""")', he_prompt, re.DOTALL)
    if doc_match:
        return doc_match.group(1)
    doc_match = re.search(r"(    '''.*?''')", he_prompt, re.DOTALL)
    if doc_match:
        return doc_match.group(1)
    return ''


def build_solution(he_prompt: str, body_lines: list) -> str:
    """
    Build final solution:
      imports
      def signature(args):
          docstring  (preserved for context, won't affect execution)
          [generated body]
    """
    header = get_clean_prompt_header(he_prompt)
    docstring = get_docstring(he_prompt)

    indented = []
    for line in body_lines:
        if line.strip() == '':
            indented.append('')
        elif line.startswith(('    ', '\t')):
            indented.append(line)
        else:
            indented.append('    ' + line)

    parts = [header]
    if docstring:
        parts.append(docstring)
    parts.append('\n'.join(indented))

    return '\n'.join(parts) + '\n'


def make_model_prompt(he_prompt: str) -> str:
    """Minimal SFT-format prompt for the model."""
    doc_match = re.search(r'"""(.*?)"""', he_prompt, re.DOTALL)
    if not doc_match:
        doc_match = re.search(r"'''(.*?)'''", he_prompt, re.DOTALL)

    if doc_match:
        raw = doc_match.group(1).strip()
        lines = [l.strip() for l in raw.split('\n')
                 if l.strip() and not l.strip().startswith('>>>')]
        desc = lines[0][:120] if lines else 'solve this'
    else:
        m = re.search(r'def (\w+)', he_prompt)
        desc = m.group(1).replace('_', ' ') if m else 'solve this'

    sig = ''
    for line in he_prompt.split('\n'):
        if line.strip().startswith('def ') and not line.startswith(' '):
            sig = line.rstrip()
            if not sig.endswith(':'):
                sig += ':'
            break

    return f"# Task: {desc}\n\n{sig}\n"


def execute_solution(solution: str, test_code: str, timeout: int = 5) -> bool:
    full = solution + "\n\n" + test_code + "\n\ncheck(candidate)"
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def run_diagnostic(model, tokenizer, problems, device, n=5):
    print("\n--- Diagnostic ---")
    for task_id in sorted(problems.keys())[:n]:
        p = problems[task_id]
        he_prompt = p["prompt"]
        test_code = p.get("test", "")
        model_p = make_model_prompt(he_prompt)
        raw = generate_body(model, tokenizer, model_p, device)
        body = extract_body_lines(raw)
        solution = build_solution(he_prompt, body)
        ok = execute_solution(solution, test_code)

        print(f"\n{task_id}  -->  {'PASS ✓' if ok else 'FAIL'}")
        print(f"  Body      : {body[:3]}")
        print(f"  Solution  :\n{solution[:200]}")
        print("-" * 40)


def run_humaneval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PyCraft-1 HumanEval Benchmark")
    print("=" * 60)

    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        print(f"Loaded {len(problems)} HumanEval+ problems")
    except Exception:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test",
                          trust_remote_code=True)
        problems = {row["task_id"]: row for row in ds}
        print(f"Loaded {len(problems)} HumanEval problems")

    print("Loading model...")
    model, tokenizer = load_model(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    run_diagnostic(model, tokenizer, problems, device, n=5)

    print("\nProceed with full evaluation? (y/n): ", end="")
    if input().strip().lower() != 'y':
        print("Cancelled.")
        return

    task_ids = sorted(problems.keys())
    results = []
    passed = failed = 0
    t0 = time.time()

    print(f"\nEvaluating {len(task_ids)} problems...\n")

    for i, task_id in enumerate(task_ids):
        p = problems[task_id]
        he_prompt = p["prompt"]
        test_code = p.get("test", "")

        model_p = make_model_prompt(he_prompt)
        raw = generate_body(model, tokenizer, model_p, device)
        body = extract_body_lines(raw)
        solution = build_solution(he_prompt, body)
        ok = execute_solution(solution, test_code)

        if ok:
            passed += 1
        else:
            failed += 1

        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (len(task_ids) - i - 1)
        print(f"  [{i+1:>3}/{len(task_ids)}] {task_id:<25} "
              f"{'PASS' if ok else 'FAIL'}  "
              f"(~{remaining/60:.0f}m left)")

        results.append({"task_id": task_id, "solution": solution})

    pass_at_1 = 100.0 * passed / len(task_ids)

    print()
    print("=" * 60)
    print("PyCraft-1 HumanEval Results")
    print("=" * 60)
    print(f"  Pass@1          : {pass_at_1:.2f}%")
    print(f"  Passed          : {passed} / {len(task_ids)}")
    print()
    print("  Comparable models:")
    print(f"  GPT-Neo 125M    :  0.83%  (300B tokens, multi-GPU)")
    print(
        f"  PyCraft-1 (ours): {pass_at_1:>5.2f}%  (1.05B tokens, 1x RTX 3050)")
    print(f"  CodeParrot 110M :  3.80%  (50B tokens, multi-GPU)")

    jsonl_path = RESULTS_DIR / "pycraft1_humaneval.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"task_id": r["task_id"],
                                "solution": r["solution"]}) + "\n")

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": "PyCraft-1", "pass_at_1": round(pass_at_1, 4),
                   "passed": passed, "total": len(task_ids)}, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    print("=" * 60)
    return pass_at_1


if __name__ == "__main__":
    run_humaneval()
