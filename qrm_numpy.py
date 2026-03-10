"""
EKSPERYMENT QRM - wersja NumPy-only
====================================
Hipoteza: QRM interference bias przyspiesza konwergencje
          w porownaniu ze standardowa uwaga dot-product.

Zero zewnetrznych zaleznosci - dziala na czystym Python + NumPy.
Uruchomienie: python3 qrm_numpy.py

Architektura:
  Mini-transformer (2 warstwy, 4 glowice, d=64)
  Korpus: syntetyczny tekst z powtarzajacymi sie wzorcami fazowymi
  Metryka: cross-entropy loss co N krokow
"""

import numpy as np
import math
import time

np.random.seed(42)

# -------------------------------------------------------------
#  KONFIGURACJA
# -------------------------------------------------------------
VOCAB_SIZE  = 64    # maly slownik - wzorce sa wazniejsze niz bogactwo
D_MODEL     = 64    # wymiar embeddingow
N_HEADS     = 4     # glowice uwagi
N_LAYERS    = 2     # warstwy transformera
SEQ_LEN     = 16    # dlugosc sekwencji
BATCH       = 8     # batch size
STEPS       = 300   # kroki treningu
EVAL_EVERY  = 30    # co ile krokow ewaluacja
LR          = 0.003 # learning rate
D_HEAD      = D_MODEL // N_HEADS  # 16

# QRM
DIM_ANGLES  = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])


# -------------------------------------------------------------
#  POMOCNICZE FUNKCJE
# -------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def cross_entropy(logits, targets):
    """logits: (batch, seq, vocab), targets: (batch, seq)"""
    b, s, v = logits.shape
    logits_flat = logits.reshape(-1, v)
    targets_flat = targets.reshape(-1)
    probs = softmax(logits_flat)
    eps   = 1e-9
    loss  = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + eps)
    return loss.mean()


# -------------------------------------------------------------
#  QRM INTERFERENCE
# -------------------------------------------------------------

def semantic_phase(hidden, W_qrm):
    """
    Faza semantyczna z hidden states.
    hidden:  (batch, seq, d_model)
    W_qrm:  (d_model, 4)  - zamrozony
    return: (batch, seq)
    """
    proj  = hidden @ W_qrm                          # (batch, seq, 4)
    mags  = np.abs(proj)
    total = mags.sum(axis=-1, keepdims=True) + 1e-9
    phase = (mags * DIM_ANGLES).sum(axis=-1) / total.squeeze(-1)
    return phase                                    # (batch, seq)

def qrm_interference(hidden, W_qrm, lam):
    """
    Macierz interferencji fazowej (causal).
    return: (batch, seq, seq) - bias do logitow uwagi
    """
    phases = semantic_phase(hidden, W_qrm)          # (batch, seq)
    phi_i  = phases[:, :, np.newaxis]               # (batch, seq, 1)
    phi_j  = phases[:, np.newaxis, :]               # (batch, 1, seq)
    interf = np.cos(phi_i - phi_j)                  # (batch, seq, seq)

    # Causal mask
    seq    = hidden.shape[1]
    causal = np.tril(np.ones((seq, seq)))
    interf = interf * causal

    return lam * interf                             # (batch, seq, seq)


# -------------------------------------------------------------
#  MINI-TRANSFORMER
# -------------------------------------------------------------

class MiniTransformer:
    """
    Prosty transformer z recznym forward i backprop przez numeryczny gradient.
    Parametry przechowywane jako slownik numpy array.
    """
    def __init__(self, use_qrm=False):
        self.use_qrm = use_qrm
        self.params  = {}
        self._init_params()

    def _init_params(self):
        p = self.params
        scale = 0.02

        # Embedding
        p['embed'] = np.random.randn(VOCAB_SIZE, D_MODEL) * scale
        p['pos']   = np.random.randn(SEQ_LEN, D_MODEL) * scale

        for i in range(N_LAYERS):
            # Uwaga: Q, K, V, O projections
            p[f'Wq{i}'] = np.random.randn(D_MODEL, D_MODEL) * scale
            p[f'Wk{i}'] = np.random.randn(D_MODEL, D_MODEL) * scale
            p[f'Wv{i}'] = np.random.randn(D_MODEL, D_MODEL) * scale
            p[f'Wo{i}'] = np.random.randn(D_MODEL, D_MODEL) * scale

            # FFN
            p[f'W1{i}'] = np.random.randn(D_MODEL, D_MODEL * 4) * scale
            p[f'b1{i}'] = np.zeros(D_MODEL * 4)
            p[f'W2{i}'] = np.random.randn(D_MODEL * 4, D_MODEL) * scale
            p[f'b2{i}'] = np.zeros(D_MODEL)

        # LM head
        p['lm_head'] = np.random.randn(D_MODEL, VOCAB_SIZE) * scale

        # QRM - zamrozony W_qrm, uczacy sie lam
        if self.use_qrm:
            W = np.random.randn(D_MODEL, 4)
            # Ortogonalizacja (QR)
            Q, _ = np.linalg.qr(W)
            self.W_qrm = Q[:, :4]    # zamrozony
            self.lam   = 0.1          # uczacy sie (skalar)

    def attention(self, x, layer_idx):
        """
        Multi-head attention z opcjonalnym QRM bias.
        x: (batch, seq, d_model)
        """
        B, S, D = x.shape
        i = layer_idx
        p = self.params

        Q = x @ p[f'Wq{i}']    # (B, S, D)
        K = x @ p[f'Wk{i}']
        V = x @ p[f'Wv{i}']

        # Reshape do glowic
        Q = Q.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)  # (B,H,S,dh)
        K = K.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)

        # Logity uwagi
        scale  = math.sqrt(D_HEAD)
        logits = Q @ K.transpose(0, 1, 3, 2) / scale  # (B,H,S,S)

        # Causal mask
        causal = np.tril(np.ones((S, S)))
        logits = logits + (1 - causal) * (-1e9)

        # QRM bias - dodany do logitow przed softmax
        if self.use_qrm:
            bias = qrm_interference(x, self.W_qrm, self.lam)  # (B,S,S)
            logits = logits + bias[:, np.newaxis, :, :]        # broadcast na glowice

        attn = softmax(logits)           # (B,H,S,S)
        out  = attn @ V                  # (B,H,S,dh)
        out  = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out  = out @ p[f'Wo{i}']
        return out

    def forward(self, tokens):
        """
        tokens: (batch, seq) int
        return: logits (batch, seq, vocab)
        """
        p = self.params
        x = p['embed'][tokens] + p['pos'][np.newaxis, :tokens.shape[1], :]

        for i in range(N_LAYERS):
            # Attention + residual
            x = layer_norm(x + self.attention(x, i))
            # FFN + residual
            h = gelu(x @ p[f'W1{i}'] + p[f'b1{i}'])
            x = layer_norm(x + (h @ p[f'W2{i}'] + p[f'b2{i}']))

        return x @ p['lm_head']

    def loss(self, tokens):
        x_in  = tokens[:, :-1]
        x_out = tokens[:, 1:]
        logits = self.forward(x_in)
        return cross_entropy(logits, x_out)

    def num_params(self):
        n = sum(v.size for v in self.params.values())
        if self.use_qrm:
            n += 1  # lam
        return n


# -------------------------------------------------------------
#  NUMERYCZNY GRADIENT (SGD krok)
# -------------------------------------------------------------

def numerical_grad_step(model, tokens, lr, eps=1e-4):
    """
    Gradient numeryczny dla kazdego parametru.
    Wolniejszy niz backprop ale dziala bez torch/autograd.
    Uzywamy go dla lam QRM - reszta przez uproszczony backprop.
    """
    # Uproszczony krok: perturbacja lam
    if model.use_qrm:
        base_loss = model.loss(tokens)
        model.lam += eps
        loss_plus  = model.loss(tokens)
        model.lam -= 2 * eps
        loss_minus = model.loss(tokens)
        model.lam += eps  # przywroc
        grad_lam   = (loss_plus - loss_minus) / (2 * eps)
        model.lam -= lr * grad_lam

    # Dla pozostalych parametrow - uproszczony SGD z perturbacja
    # (dla szybkosci robimy losowy podzbior parametrow per krok)
    base_loss = model.loss(tokens)
    for key in list(model.params.keys()):
        param = model.params[key]
        # Losowa perturbacja kierunkowa (SPSA)
        delta = np.random.randn(*param.shape)
        delta /= (np.linalg.norm(delta) + 1e-9)

        model.params[key] = param + eps * delta
        loss_plus = model.loss(tokens)
        model.params[key] = param - eps * delta
        loss_minus = model.loss(tokens)
        model.params[key] = param  # przywroc

        grad_est = (loss_plus - loss_minus) / (2 * eps) * delta
        model.params[key] = param - lr * grad_est

    return base_loss


# -------------------------------------------------------------
#  DANE SYNTETYCZNE
#  Wzorce fazowe: sekwencje maja wewnetrzna strukture
#  (token A zawsze poprzedza token B w pewnym kontekscie)
# -------------------------------------------------------------

def generate_data(n_samples=200):
    """
    Generuje sekwencje z powtarzajacymi sie wzorcami.
    System z dobrymi BC powinien nauczyc sie tych wzorcow szybciej.
    """
    data = []

    # Wzorzec 1: sekwencje rosnace (mod VOCAB)
    for _ in range(n_samples // 4):
        start = np.random.randint(0, VOCAB_SIZE - SEQ_LEN - 1)
        seq   = np.arange(start, start + SEQ_LEN + 1) % VOCAB_SIZE
        data.append(seq)

    # Wzorzec 2: sekwencje z powtorzeniem co 4
    for _ in range(n_samples // 4):
        base = np.random.randint(0, VOCAB_SIZE // 4, size=4)
        seq  = np.tile(base, (SEQ_LEN + 1) // 4 + 1)[:SEQ_LEN + 1]
        data.append(seq)

    # Wzorzec 3: sekwencje palindromiczne
    for _ in range(n_samples // 4):
        half = np.random.randint(0, VOCAB_SIZE, size=SEQ_LEN // 2 + 1)
        seq  = np.concatenate([half, half[::-1]])[:SEQ_LEN + 1]
        data.append(seq)

    # Wzorzec 4: losowe (szum - trudne)
    for _ in range(n_samples // 4):
        seq = np.random.randint(0, VOCAB_SIZE, size=SEQ_LEN + 1)
        data.append(seq)

    data = np.array(data)
    np.random.shuffle(data)
    split = int(len(data) * 0.8)
    return data[:split], data[split:]


def get_batch(data):
    idx = np.random.randint(0, len(data), size=BATCH)
    return data[idx]


# -------------------------------------------------------------
#  TRENING
# -------------------------------------------------------------

def evaluate(model, val_data, n_batches=10):
    losses = [model.loss(get_batch(val_data)) for _ in range(n_batches)]
    return math.exp(np.mean(losses))


def train(model, train_data, val_data, label):
    history = {"steps": [], "val_ppl": []}
    t0      = time.time()

    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  parametry: {model.num_params():,}")
    if model.use_qrm:
        print(f"  lam startowe: {model.lam:.4f}  (W_qrm zamrozony)")
    print(f"{'='*58}")
    print(f"  {'krok':>6}  {'val_ppl':>10}  {'czas[s]':>9}"
          + ("  {'lam':>8}" if model.use_qrm else ""))
    print(f"  {'-'*45}")

    for step in range(1, STEPS + 1):
        batch = get_batch(train_data)
        numerical_grad_step(model, batch, lr=LR)

        if step % EVAL_EVERY == 0:
            ppl     = evaluate(model, val_data)
            elapsed = time.time() - t0
            history["steps"].append(step)
            history["val_ppl"].append(ppl)

            lam_str = f"  lam={model.lam:.5f}" if model.use_qrm else ""
            print(f"  {step:>6}  {ppl:>10.3f}  {elapsed:>9.1f}s{lam_str}")

    return history


# -------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------

def main():
    print("QRM EXPERIMENT - NumPy only")
    print(f"d_model={D_MODEL}  seq={SEQ_LEN}  vocab={VOCAB_SIZE}"
          f"  layers={N_LAYERS}  heads={N_HEADS}\n")

    train_data, val_data = generate_data(n_samples=400)
    print(f"Dane: {len(train_data)} train / {len(val_data)} val sekwencji")

    # Baseline
    model_base = MiniTransformer(use_qrm=False)
    hist_base  = train(model_base, train_data, val_data,
                       label="BASELINE - standardowa uwaga dot-product")

    # QRM
    model_qrm = MiniTransformer(use_qrm=True)
    hist_qrm  = train(model_qrm, train_data, val_data,
                      label="QRM - uwaga + interference bias (lam uczacy sie)")

    # Porownanie
    print(f"\n{'='*58}")
    print(f"  WYNIKI POROWNAWCZE")
    print(f"{'='*58}")
    print(f"  {'krok':>6}  {'baseline':>10}  {'QRM':>10}  {'delta%':>9}  {'lepszy':>8}")
    print(f"  {'-'*52}")

    wins_qrm  = 0
    wins_base = 0
    for i, step in enumerate(hist_base["steps"]):
        b = hist_base["val_ppl"][i]
        q = hist_qrm["val_ppl"][i]
        d = (q - b) / b * 100
        w = "QRM [OK]" if q < b else "BASE"
        if q < b: wins_qrm  += 1
        else:     wins_base += 1
        print(f"  {step:>6}  {b:>10.3f}  {q:>10.3f}  {d:>+9.2f}%  {w:>8}")

    final_b = hist_base["val_ppl"][-1]
    final_q = hist_qrm["val_ppl"][-1]
    delta   = (final_q - final_b) / final_b * 100

    print(f"\n  Punkty wygranych: QRM={wins_qrm}  BASE={wins_base}")
    print(f"  Koncowy ppl baseline : {final_b:.3f}")
    print(f"  Koncowy ppl QRM      : {final_q:.3f}")
    print(f"  delta koncowy            : {delta:+.2f}%")
    print(f"  lam koncowe            : {model_qrm.lam:.5f}")

    print(f"\n  WNIOSEK:")
    if wins_qrm > wins_base and final_q < final_b:
        print("  [OK] HIPOTEZA POTWIERDZONA")
        print("    QRM bias przyspiesza konwergencje.")
        print("    Warunki brzegowe niosa strukturalna informacje.")
    elif abs(delta) < 3.0:
        print("  [[~]] WYNIK NEUTRALNY (delta < 3%)")
        print("    QRM nie przeszkadza. Dostroic lam lub punkt wpiecia.")
    else:
        print("  [!] HIPOTEZA ODRZUCONA w tej konfiguracji")
        print("    Sprawdzic mapowanie fazy lub architekture biastu.")


if __name__ == "__main__":
    main()
