import torch
from safetensors.torch import load_file
from model.config import get_config_120m
from model.pycraft_model import PyCraftModel
from tokenizer.tokenizer_utils import PyCraftTokenizer

device    = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PyCraftTokenizer()
cfg       = get_config_120m()
cfg.vocab_size = tokenizer.vocab_size
cfg.dropout    = 0.0

model = PyCraftModel(cfg).to(device)
model.load_state_dict(load_file('checkpoints/sft_stage1/model.safetensors', device=device))
model.eval()

from evalplus.data import get_human_eval_plus
problems = get_human_eval_plus()
task_id  = 'HumanEval/0'
prompt   = problems[task_id]['prompt']

print('=== PROMPT ===')
print(prompt)
print()

ids = tokenizer.encode(prompt)
inp = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    out = model.generate(inp, max_new_tokens=150, temperature=0.2, top_k=20)

new_ids    = out[0, len(ids):].tolist()
completion = tokenizer.decode(new_ids, skip_special_tokens=True)

print('=== RAW COMPLETION (repr) ===')
print(repr(completion))
print()
print('=== RENDERED ===')
print(completion)
