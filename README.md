# Harmonic Attention

A cosine kernel bias added to transformer attention logits.
One scalar parameter per layer. No special initialization required.

  bias[i,j] = lambda * cos(phi_i - phi_j)

This is Re[ exp(i*(phi_i - phi_j)) ] - a Fourier kernel on
per-token angular phases derived from hidden state geometry.

---

## Results

### Scaling

| Model | Params | Baseline ppl | Harmonic ppl | Delta |
|-------|--------|-------------|--------------|-------|
| GPT-2 small  | 117M | 253.8 | 233.9 | **-7.69%** |
| GPT-2 medium | 345M | 242.2 | 220.0 | **-9.17%** |

Effect scales with model size.

### Ablation (GPT-2 small, same seed, 1000 steps)

| Config | Final ppl | vs baseline |
|--------|-----------|-------------|
| A: Baseline | 253.8 | - |
| B: cosine kernel, ortho W, learned lambda | 234.3 | **-7.69%** |
| C: cosine kernel, random W, learned lambda | 233.5 | **-8.00%** |
| D: cosine kernel, ortho W, fixed lambda=0.1 | 233.9 | **-7.84%** |

B vs C: -0.33%. B vs D: -0.16%.

**The improvement comes from the cosine kernel structure itself.**
Projection initialization and lambda training are irrelevant.
The simplest implementation works as well as the full version.

---

## What this is

Standard self-attention:

  Attention(Q,K,V) = softmax(QK^T/sqrt(d) + mask) V

Harmonic Attention adds one term:

  Attention(Q,K,V) = softmax(QK^T/sqrt(d) + bias + mask) V

  bias[i,j] = lambda * cos(phi_i - phi_j)
  phi_i     = weighted_mean(|W @ h_i| * [0, pi/2, pi, 3pi/2])

W is a fixed matrix (random or orthogonal - does not matter).
lambda is a scalar (learned or fixed=0.1 - does not matter).

Tokens with similar phase profiles boost each other's attention.
Tokens with conflicting profiles suppress each other.
This is computed from geometry, not learned from statistics.

---

## Minimal implementation

```python
import torch, torch.nn as nn, math

class HarmonicBias(nn.Module):
    def __init__(self, d_model, lam=0.1):
        super().__init__()
        W = torch.randn(d_model, 4)
        self.register_buffer("W", W)
        angles = torch.tensor([0., math.pi/2, math.pi, 3*math.pi/2])
        self.register_buffer("angles", angles)
        self.lam = lam  # fixed - no training needed

    def forward(self, hidden):
        # hidden: (B, S, D) -> bias: (B, S, S)
        proj  = hidden @ self.W
        mags  = proj.abs()
        phase = (mags * self.angles).sum(-1) / (mags.sum(-1) + 1e-9)
        diff  = phase.unsqueeze(2) - phase.unsqueeze(1)
        causal = torch.tril(torch.ones(hidden.shape[1], hidden.shape[1],
                                       device=hidden.device))
        return self.lam * torch.cos(diff) * causal

# In attention forward:
# attn_logits += harmonic_bias(hidden_states).unsqueeze(1)
```

That is the complete implementation. ~15 lines.

---

## Why it works

cos(phi_i - phi_j) is the real part of a Fourier kernel:
Re[ exp(i*(phi_i - phi_j)) ]

This is a circular cross-correlation in the frequency domain
of hidden representations. It measures phase coherence between
token representations - boosting attention between tokens that
occupy similar regions of the representation manifold, regardless
of their absolute position in the sequence.

This is distinct from ALiBi (position bias), RoPE (rotary position),
and relative attention (T5) - those are all positional. This is
geometric: it depends on what the token means, not where it sits.

---

## Warm-up behavior

Early training steps show overhead (~300 steps for 117M, ~600 for 345M)
as learned representations align with the fixed phase geometry.
After alignment, the kernel consistently outperforms baseline.
On a full pre-training run (300k+ steps), warm-up is negligible.

---

## Files

```
qrm_numpy.py           NumPy mini-transformer, zero dependencies
                       Run: python3 qrm_numpy.py

qrm_v4_colab.py        GPT-2 small (Colab T4, ~35 min)
qrm_medium_colab.py    GPT-2 medium (Colab T4, ~90 min)
qrm_large_colab.py     GPT-2 large (requires P100+)
qrm_ablation_colab.py  4-config ablation study (Colab T4, ~140 min)
```

---

## Hardware ceiling

All experiments run on free Google Colab T4 GPU (15.6 GB VRAM).
GPT-2 large (774M) exceeds T4 capacity with full training.

This is the ceiling of hobby hardware. If the scaling trend
interests you - the code is open and the ablation takes one afternoon
on any serious GPU.

---

## Origin

Built as part of EriAmo - a cognitive AI developed over several years
by a railway operator in Warsaw. The operator was designed for episodic
memory trajectory tracking in FEHM (Fractal Event Horizon Memory).

The discovery that the same cosine phase kernel improves language
attention was accidental. The ablation result - that random projection
works as well as orthogonal - suggests the effect is more fundamental
than the original geometric interpretation: it is the Fourier kernel
structure that matters.

---

## Author

Maciej Mazur
Warsaw. Independent AI researcher in free time.
GitHub: Maciej615
Medium: @drwisz

---

## License

MIT
