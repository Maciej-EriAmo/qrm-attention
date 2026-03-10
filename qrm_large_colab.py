# QRM Experiment - GPT-2 large (774M)
# Runtime: ~3h on T4 GPU (free Colab - may hit time limit)
# Recommendation: use Colab Pro or Kaggle P100 (30h/week free)

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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

MODEL_NAME = "gpt2-large"    # 774M params
MAX_STEPS  = 1000
EVAL_EVERY = 100
BATCH_SIZE = 1               # large needs small batch - 774M is tight on T4
SEQ_LEN    = 64
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

# -- VRAM CHECK -------------------------------------------------------------
def check_vram():
    if DEVICE != "cuda":
        return True
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM available: {vram_gb:.1f} GB")
    if vram_gb < 14:
        print("WARNING: less than 14GB VRAM - large model may OOM")
        print("Recommendation: use Kaggle P100 or Colab Pro A100")
    return True

check_vram()

# -- SANITY CHECK -----------------------------------------------------------
print("\nSanity check...")
torch.manual_seed(0)
_m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
_m = patch_gpt2_with_qrm(_m)
_x = torch.randint(0, 100, (1, 16)).to(DEVICE)
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
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(x, labels=y).loss
        loss.backward()
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

            # Save checkpoint every 500 steps - in case Colab disconnects
            if step % 500 == 0:
                torch.save(hist, f"hist_{label[:4].strip()}_{step}.pt")
                print(f"  [checkpoint saved at step {step}]")

    return hist

# -- RUN --------------------------------------------------------------------
torch.manual_seed(42)
model_base = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model_base.gradient_checkpointing_enable()
n_base = sum(p.numel() for p in model_base.parameters())
print(f"Baseline params: {n_base:,}")
hist_base = train_model(model_base, "BASELINE: GPT-2 large standard attention")
del model_base
torch.cuda.empty_cache()
print("Baseline done. Clearing VRAM...\n")

torch.manual_seed(42)
model_qrm = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model_qrm.gradient_checkpointing_enable()
model_qrm = patch_gpt2_with_qrm(model_qrm)
n_qrm = sum(p.numel() for p in model_qrm.parameters())
print(f"QRM params: {n_qrm:,}  (+{n_qrm - n_base} vs baseline)")
hist_qrm = train_model(model_qrm, "QRM: GPT-2 large + interference bias v4")

# -- RESULTS ----------------------------------------------------------------
print(f"\n{'='*62}")
print("  RESULTS - GPT-2 large (774M)")
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

print(f"\n  SCALING TABLE:")
print(f"  {'Model':<20} {'Params':>8}  {'Final ppl BASE':>14}  {'Final ppl QRM':>13}  {'Delta':>8}  {'Wins':>6}")
print(f"  {'-'*78}")
print(f"  {'GPT-2 small':<20} {'117M':>8}  {254.063:>14.3f}  {233.934:>13.3f}  {-7.92:>+7.2f}%  {'6/10':>6}")
print(f"  {'GPT-2 medium':<20} {'345M':>8}  {242.233:>14.3f}  {220.017:>13.3f}  {-9.17:>+7.2f}%  {'2/10':>6}")
print(f"  {'GPT-2 large':<20} {'774M':>8}  {fb:>14.3f}  {fq:>13.3f}  {fd:>+7.2f}%  {f'{wins}/10':>6}")

# -- PLOT -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
steps = hist_base["steps"]

axes[0].plot(steps, hist_base["val_ppl"], "b-o", label="Baseline GPT-2 large", lw=2)
axes[0].plot(steps, hist_qrm["val_ppl"],  "r-o", label="GPT-2 large + QRM bias", lw=2)
axes[0].set_xlabel("Training steps")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Convergence: Baseline vs QRM | GPT-2 large (774M)")
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
    f"QRM v4 | gpt2-large (774M) | WikiText-2 | "
    f"delta: {fd:+.2f}% | lam={np.mean(lams):.5f}",
    y=1.02
)
plt.tight_layout()
plt.savefig("qrm_results_large.png", dpi=150, bbox_inches="tight")
plt.show()

# -- SCALING PLOT -----------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 5))
models   = ["GPT-2 small\n(117M)", "GPT-2 medium\n(345M)", "GPT-2 large\n(774M)"]
params   = [117, 345, 774]
deltas_s = [-7.92, -9.17, fd]

ax.plot(params, deltas_s, "go-", lw=2, markersize=10)
ax.axhline(0, color="black", lw=1, linestyle="--")
for i, (p, d, m) in enumerate(zip(params, deltas_s, models)):
    ax.annotate(f"{d:+.2f}%", (p, d),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(params)
ax.set_xticklabels(models)
ax.set_xlabel("Model size")
ax.set_ylabel("Delta perplexity vs baseline (%)")
ax.set_title("QRM effect scales with model size")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("qrm_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: qrm_scaling.png")

# -- SAVE -------------------------------------------------------------------
with open("qrm_results_large.csv", "w", newline="") as f:
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
    files.download("qrm_results_large.png")
    files.download("qrm_results_large.csv")
    files.download("qrm_scaling.png")
    print("Download started.")
except Exception:
    print("Saved: qrm_results_large.png / qrm_results_large.csv / qrm_scaling.png")
