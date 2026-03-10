# QRM Experiment - GPT-2 medium (345M)
# Same v4 architecture as gpt2 small run.
# Runtime: ~90 min on T4 GPU (free Colab)

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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODEL_NAME = "gpt2-medium"   # 345M params
MAX_STEPS  = 1000
EVAL_EVERY = 100
BATCH_SIZE = 2               # smaller batch - medium is bigger
SEQ_LEN    = 128
LR         = 3e-4

# -- QRM MODULE -------------------------------------------------------------
class QRMInterference(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        W = torch.empty(d_model, 4)
        nn.init.orthogonal_(W)
        self.register_buffer("W_qrm", W)
        angles = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2])
        self.register_buffer("dim_angles", angles)
        self.lam = nn.Parameter(torch.tensor(0.1))

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
        return self.lam * interf * causal

# -- ATTENTION FORWARD REPLACEMENT ------------------------------------------
def make_qrm_attention_forward(qrm_module, n_heads, d_head):
    def qrm_attention_forward(
        self,
        hidden_states,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        **kwargs
    ):
        B, S, D = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(D, dim=2)

        def split_heads(x):
            return x.view(B, S, n_heads, d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

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

        # QRM bias - direct logit injection
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


def patch_gpt2_with_qrm(model):
    d_model = model.config.n_embd
    n_layer = model.config.n_layer
    n_heads = model.config.n_head
    d_head  = d_model // n_heads
    dev     = next(model.parameters()).device

    model.qrm_modules = nn.ModuleList([
        QRMInterference(d_model).to(dev) for _ in range(n_layer)
    ])

    for i, block in enumerate(model.transformer.h):
        qrm     = model.qrm_modules[i]
        new_fwd = make_qrm_attention_forward(qrm, n_heads, d_head)
        block.attn.forward = types.MethodType(new_fwd, block.attn)

    n_lrn = sum(p.numel() for p in model.qrm_modules.parameters()
                if p.requires_grad)
    print(f"QRM learnable params: {n_lrn}  (lam x{n_layer}, W_qrm frozen)")
    return model

# -- SANITY CHECK -----------------------------------------------------------
print("Sanity check...")
torch.manual_seed(0)
_m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
_m = patch_gpt2_with_qrm(_m)
_x = torch.randint(0, 100, (2, 16)).to(DEVICE)
_out = _m(_x, labels=_x)
print(f"Loss: {_out.loss.item():.4f}")
_out.loss.backward()
lam_grad = _m.qrm_modules[0].lam.grad.item()
print(f"lam grad: {lam_grad:.6f}  <- must be non-zero")
assert abs(lam_grad) > 1e-8, "ERROR: gradient not flowing!"
del _m
torch.cuda.empty_cache()
print("Sanity check PASSED.\n")

# -- DATA -------------------------------------------------------------------
print("Loading WikiText-2...")
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

    print(f"\n{'-'*62}")
    print(f"  {label}")
    print(f"{'-'*62}")
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
                lv = [m.lam.item() for m in model.qrm_modules]
                lam_str = f"  lam={np.mean(lv):.5f} [{min(lv):.4f}..{max(lv):.4f}]"
            print(f"  {step:>6}  {ppl:>10.3f}  {time.time()-t0:>7.1f}s{lam_str}")

    return hist

# -- RUN --------------------------------------------------------------------
torch.manual_seed(42)
model_base = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
n_base = sum(p.numel() for p in model_base.parameters())
print(f"Baseline params: {n_base:,}")
hist_base = train_model(model_base, "BASELINE: GPT-2 medium standard attention")

torch.manual_seed(42)
model_qrm = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model_qrm = patch_gpt2_with_qrm(model_qrm)
n_qrm = sum(p.numel() for p in model_qrm.parameters())
print(f"QRM params: {n_qrm:,}  (+{n_qrm - n_base} vs baseline)")
hist_qrm = train_model(model_qrm, "QRM: GPT-2 medium + interference bias v4")

# -- RESULTS ----------------------------------------------------------------
print(f"\n{'='*62}")
print("  RESULTS - GPT-2 medium (345M)")
print(f"{'='*62}")
print(f"  {'step':>6}  {'baseline':>10}  {'QRM':>10}  {'delta%':>9}  winner")
print(f"  {'-'*58}")

wins, deltas = 0, []
for i, step in enumerate(hist_base["steps"]):
    b = hist_base["val_ppl"][i]
    q = hist_qrm["val_ppl"][i]
    d = (q - b) / b * 100
    deltas.append(d)
    w = "QRM [OK]" if q < b else "BASE"
    if q < b: wins += 1
    print(f"  {step:>6}  {b:>10.3f}  {q:>10.3f}  {d:>+9.2f}%  {w}")

fb   = hist_base["val_ppl"][-1]
fq   = hist_qrm["val_ppl"][-1]
fd   = (fq - fb) / fb * 100
lams = [m.lam.item() for m in model_qrm.qrm_modules]

print(f"\n  QRM wins: {wins}/{len(hist_base['steps'])}")
print(f"  Final baseline : {fb:.3f}")
print(f"  Final QRM      : {fq:.3f}")
print(f"  Final delta    : {fd:+.2f}%")
print(f"  lam range      : {min(lams):.5f} .. {max(lams):.5f}")
print(f"  lam mean       : {np.mean(lams):.5f}")

# Compare with small model results
print(f"\n  SCALING COMPARISON:")
print(f"  {'Model':<20} {'Params':>10}  {'Final delta':>12}  {'Wins':>6}")
print(f"  {'-'*54}")
print(f"  {'GPT-2 small':<20} {'117M':>10}  {'-7.92%':>12}  {'6/10':>6}")
print(f"  {'GPT-2 medium':<20} {'345M':>10}  {fd:>+11.2f}%  {wins}/{len(hist_base['steps']):>4}")

# -- PLOT -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
steps = hist_base["steps"]

axes[0].plot(steps, hist_base["val_ppl"], "b-o", label="Baseline GPT-2 medium", lw=2)
axes[0].plot(steps, hist_qrm["val_ppl"],  "r-o", label="GPT-2 medium + QRM bias", lw=2)
axes[0].set_xlabel("Training steps")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Convergence: Baseline vs QRM | GPT-2 medium")
axes[0].legend()
axes[0].grid(alpha=0.3)

colors = ["green" if d < 0 else "red" for d in deltas]
axes[1].bar(steps, deltas, color=colors, alpha=0.7, width=max(steps)*0.06)
axes[1].axhline(0, color="black", lw=1)
axes[1].set_xlabel("Training steps")
axes[1].set_ylabel("Delta perplexity (%)")
axes[1].set_title("QRM improvement (negative = better)")
axes[1].grid(alpha=0.3, axis="y")

plt.suptitle(
    f"QRM v4 | gpt2-medium (345M) | WikiText-2 | "
    f"delta: {fd:+.2f}% | lam={np.mean(lams):.5f}",
    y=1.02
)
plt.tight_layout()
plt.savefig("qrm_results_medium.png", dpi=150, bbox_inches="tight")
plt.show()

# -- SAVE -------------------------------------------------------------------
with open("qrm_results_medium.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step","baseline_ppl","qrm_ppl","delta_pct","lam_mean"])
    for i, step in enumerate(hist_base["steps"]):
        w.writerow([step,
                    round(hist_base["val_ppl"][i], 4),
                    round(hist_qrm["val_ppl"][i],  4),
                    round(deltas[i], 4),
                    round(np.mean(lams), 6)])

try:
    from google.colab import files
    files.download("qrm_results_medium.png")
    files.download("qrm_results_medium.csv")
    print("Download started.")
except Exception:
    print("Saved: qrm_results_medium.png / qrm_results_medium.csv")
