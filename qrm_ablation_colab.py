# QRM Ablation Study - GPT-2 small, WikiText-2
# Tests 4 configurations to answer: "is this just regularization?"
#
# Config A: baseline         (lam=0, no bias)
# Config B: QRM ortho        (lam learnable, W_qrm orthogonal) <- main result
# Config C: QRM random       (lam learnable, W_qrm random)     <- ablation 1
# Config D: QRM fixed lam    (lam=0.1 fixed, W_qrm orthogonal) <- ablation 2
#
# If B > C: orthogonal geometry matters, not just regularization
# If B > D: lambda learning matters, not just fixed bias
# If B > A: the whole thing works (already known)

import subprocess
subprocess.run(["pip", "install", "transformers==4.40.0", "datasets", "torch", "-q"])

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, csv
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import types

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME = "gpt2"
MAX_STEPS  = 1000
EVAL_EVERY = 100
BATCH_SIZE = 4
SEQ_LEN    = 128
LR         = 3e-4

# -- QRM MODULE (parametric) ------------------------------------------------
class QRMInterference(nn.Module):
    def __init__(self, d_model, ortho=True, fixed_lam=None):
        super().__init__()
        W = torch.empty(d_model, 4)
        if ortho:
            nn.init.orthogonal_(W)   # structured: 4 orthogonal axes
        else:
            nn.init.normal_(W)       # random: no structure
            W = W / (W.norm(dim=0, keepdim=True) + 1e-9)
        self.register_buffer("W_qrm", W)
        angles = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2])
        self.register_buffer("dim_angles", angles)

        if fixed_lam is not None:
            self.lam = fixed_lam          # float, not Parameter
            self.lam_learnable = False
        else:
            self.lam = nn.Parameter(torch.tensor(0.1))
            self.lam_learnable = True

    def forward(self, hidden):
        proj  = hidden @ self.W_qrm
        mags  = proj.abs()
        total = mags.sum(-1, keepdim=True) + 1e-9
        phase = (mags * self.dim_angles).sum(-1) / total.squeeze(-1)
        phi_i = phase.unsqueeze(2)
        phi_j = phase.unsqueeze(1)
        interf = torch.cos(phi_i - phi_j)
        S = hidden.shape[1]
        causal = torch.tril(torch.ones(S, S, device=hidden.device))
        lam = self.lam if not self.lam_learnable else self.lam
        return lam * interf * causal

# -- ATTENTION FORWARD REPLACEMENT ------------------------------------------
def make_qrm_attention_forward(qrm_module, n_heads, d_head):
    def qrm_attention_forward(
        self, hidden_states, past_key_values=None,
        cache_position=None, attention_mask=None, **kwargs
    ):
        B, S, D = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(D, dim=2)

        def split_heads(x):
            return x.view(B, S, n_heads, d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        scale       = math.sqrt(d_head)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / scale

        kv_len  = k.shape[-2]
        q_len   = q.shape[-2]
        causal  = torch.tril(
            torch.ones(kv_len, kv_len, device=hidden_states.device, dtype=torch.bool)
        )[-q_len:]
        mask_val    = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(~causal, mask_val)

        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask

        qrm_bias    = qrm_module(hidden_states)
        attn_logits = attn_logits + qrm_bias.unsqueeze(1)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, q_len, D)
        out = self.c_proj(context)
        out = self.resid_dropout(out)
        return out, None

    return qrm_attention_forward


def patch_model(model, ortho=True, fixed_lam=None):
    d_model = model.config.n_embd
    n_layer = model.config.n_layer
    n_heads = model.config.n_head
    d_head  = d_model // n_heads
    dev     = next(model.parameters()).device

    model.qrm_modules = nn.ModuleList([
        QRMInterference(d_model, ortho=ortho, fixed_lam=fixed_lam).to(dev)
        for _ in range(n_layer)
    ])

    for i, block in enumerate(model.transformer.h):
        qrm     = model.qrm_modules[i]
        new_fwd = make_qrm_attention_forward(qrm, n_heads, d_head)
        block.attn.forward = types.MethodType(new_fwd, block.attn)

    return model

# -- DATA -------------------------------------------------------------------
print("Loading data...")
tokenizer  = GPT2Tokenizer.from_pretrained(MODEL_NAME)
dataset    = load_dataset("wikitext", "wikitext-2-raw-v1")

def encode(split):
    text = "\n".join(dataset[split]["text"])
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)

train_data = encode("train")
val_data   = encode("validation")
print(f"Train: {len(train_data):,}  Val: {len(val_data):,} tokens")

def get_batch(data):
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x  = torch.stack([data[i:i+SEQ_LEN]     for i in ix])
    y  = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def evaluate(model, n=20):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch(val_data)
        losses.append(model(x, labels=y).loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))

# -- TRAIN ------------------------------------------------------------------
def train_model(model, label):
    opt  = torch.optim.AdamW(model.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEPS)
    hist = {"steps": [], "val_ppl": []}
    t0   = time.time()

    print(f"\n{'-'*66}")
    print(f"  {label}")
    print(f"{'-'*66}")
    print(f"  {'step':>6}  {'val_ppl':>10}  {'time':>8}")

    model.train()
    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(train_data)
        opt.zero_grad()
        model(x, labels=y).loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % EVAL_EVERY == 0:
            ppl = evaluate(model)
            hist["steps"].append(step)
            hist["val_ppl"].append(ppl)
            lam_str = ""
            if hasattr(model, "qrm_modules"):
                lv = [m.lam.item() if hasattr(m.lam, 'item') else m.lam
                      for m in model.qrm_modules]
                lam_str = f"  lam={np.mean(lv):.5f}"
            print(f"  {step:>6}  {ppl:>10.3f}  {time.time()-t0:>7.1f}s{lam_str}")

    return hist

# -- RUN ALL 4 CONFIGS ------------------------------------------------------
results = {}

# A: Baseline
print("\n[1/4] BASELINE")
torch.manual_seed(42)
m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
results["A_baseline"] = train_model(m, "A: Baseline (no bias)")
del m; torch.cuda.empty_cache()

# B: QRM orthogonal + learnable lam  <- main result
print("\n[2/4] QRM ORTHO (main)")
torch.manual_seed(42)
m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
m = patch_model(m, ortho=True, fixed_lam=None)
results["B_qrm_ortho"] = train_model(m, "B: QRM orthogonal W + learnable lam")
del m; torch.cuda.empty_cache()

# C: QRM random W + learnable lam  <- ablation: does orthogonality matter?
print("\n[3/4] QRM RANDOM W")
torch.manual_seed(42)
m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
m = patch_model(m, ortho=False, fixed_lam=None)
results["C_qrm_random"] = train_model(m, "C: QRM random W + learnable lam")
del m; torch.cuda.empty_cache()

# D: QRM orthogonal + fixed lam=0.1  <- ablation: does learning lam matter?
print("\n[4/4] QRM FIXED LAM")
torch.manual_seed(42)
m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
m = patch_model(m, ortho=True, fixed_lam=0.1)
results["D_qrm_fixed"] = train_model(m, "D: QRM orthogonal W + fixed lam=0.1")
del m; torch.cuda.empty_cache()

# -- ABLATION TABLE ---------------------------------------------------------
print(f"\n{'='*70}")
print("  ABLATION RESULTS")
print(f"{'='*70}")
print(f"  {'Config':<35} {'Final ppl':>10}  {'vs A':>8}  {'verdict'}")
print(f"  {'-'*66}")

configs = [
    ("A_baseline",  "A: Baseline"),
    ("B_qrm_ortho", "B: QRM ortho + learnable lam  [MAIN]"),
    ("C_qrm_random","C: QRM random + learnable lam [ABLATION]"),
    ("D_qrm_fixed", "D: QRM ortho + fixed lam=0.1  [ABLATION]"),
]

ppl_A = results["A_baseline"]["val_ppl"][-1]
for key, label in configs:
    ppl = results[key]["val_ppl"][-1]
    delta = (ppl - ppl_A) / ppl_A * 100
    if key == "A_baseline":
        verdict = "reference"
    elif ppl < ppl_A:
        verdict = "BETTER than baseline"
    else:
        verdict = "worse than baseline"
    print(f"  {label:<35} {ppl:>10.3f}  {delta:>+7.2f}%  {verdict}")

ppl_B = results["B_qrm_ortho"]["val_ppl"][-1]
ppl_C = results["C_qrm_random"]["val_ppl"][-1]
ppl_D = results["D_qrm_fixed"]["val_ppl"][-1]

print(f"\n  KEY QUESTIONS:")
print(f"  B vs C (ortho vs random): {(ppl_C-ppl_B)/ppl_B*100:+.2f}%  ", end="")
print("orthogonal geometry helps" if ppl_B < ppl_C else "random works equally")
print(f"  B vs D (learned vs fixed lam): {(ppl_D-ppl_B)/ppl_B*100:+.2f}%  ", end="")
print("learning lam helps" if ppl_B < ppl_D else "fixed lam works equally")

# -- PLOT -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
steps = results["A_baseline"]["steps"]
colors = {"A_baseline": "blue", "B_qrm_ortho": "red",
          "C_qrm_random": "orange", "D_qrm_fixed": "green"}
labels = {"A_baseline": "A: Baseline",
          "B_qrm_ortho": "B: QRM ortho [MAIN]",
          "C_qrm_random": "C: QRM random W",
          "D_qrm_fixed": "D: QRM fixed lam"}

for key, label in labels.items():
    axes[0].plot(steps, results[key]["val_ppl"],
                 f"{'-' if 'baseline' in key else '-'}o",
                 color=colors[key], label=label, lw=2,
                 linestyle="--" if "random" in key or "fixed" in key else "-")

axes[0].set_xlabel("Training steps")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Ablation Study: GPT-2 small | WikiText-2")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Bar chart: final ppl
final_ppls = [results[k]["val_ppl"][-1] for k, _ in configs]
config_labels = ["A\nBaseline", "B\nQRM ortho\n[MAIN]",
                 "C\nQRM random\nW", "D\nQRM ortho\nfixed lam"]
bar_colors = ["blue", "red", "orange", "green"]
bars = axes[1].bar(config_labels, final_ppls, color=bar_colors, alpha=0.75)
axes[1].set_ylabel("Final validation perplexity (lower = better)")
axes[1].set_title("Final perplexity by configuration")
for bar, ppl in zip(bars, final_ppls):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{ppl:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].grid(alpha=0.3, axis="y")

plt.suptitle("QRM Harmonic Attention - Ablation Study", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("qrm_ablation.png", dpi=150, bbox_inches="tight")
plt.show()

# -- CSV --------------------------------------------------------------------
with open("qrm_ablation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step"] + [k for k, _ in configs])
    for i, step in enumerate(steps):
        row = [step] + [round(results[k]["val_ppl"][i], 4) for k, _ in configs]
        w.writerow(row)

try:
    from google.colab import files
    files.download("qrm_ablation.png")
    files.download("qrm_ablation.csv")
    print("Download started.")
except Exception:
    print("Saved: qrm_ablation.png / qrm_ablation.csv")
