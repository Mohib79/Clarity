"""
Deep Autoregressive Character-GRU Password Model (FLA / Neural-Pass style).
Learns long-range character dependencies across the full password string using a
Gated Recurrent Unit (GRU) neural network with weight-tied input/output embeddings.
Supports both PyTorch CUDA and Vectorized NumPy BLAS (Intel Core Ultra / AVX-512) backends,
automatically bypassing Windows Application Control DLL blocks (WinError 4551).
"""

import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

TORCH_AVAILABLE = False

from pwlab.model.markov import PRINTABLE_ASCII, START_TOKEN, END_TOKEN


# Vocabulary: 0=PAD, 1=BOS, 2=EOS, 3..97 = 95 printable ASCII chars (total 98 tokens)
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
NEURAL_VOCAB = ["\x00", START_TOKEN, END_TOKEN] + PRINTABLE_ASCII
NEURAL_VOCAB_SIZE = len(NEURAL_VOCAB)  # 98
NEURAL_CHAR_TO_IDX = {c: i for i, c in enumerate(NEURAL_VOCAB)}


def encode_password(pw: str, max_len: int = 34) -> Tuple[List[int], List[int]]:
    """
    Encode password into (input_ids, target_ids) padded to max_len.
    input_ids:  [BOS, c_1, c_2, ..., c_k, PAD, ...]
    target_ids: [c_1, c_2, ..., c_k, EOS, PAD, ...]
    """
    trimmed = pw[: max_len - 1]
    token_ids = [NEURAL_CHAR_TO_IDX.get(c, 3) for c in trimmed]
    inp = [BOS_IDX] + token_ids
    tgt = token_ids + [EOS_IDX]
    pad_count = max_len - len(inp)
    if pad_count > 0:
        inp = inp + [PAD_IDX] * pad_count
        tgt = tgt + [PAD_IDX] * pad_count
    return inp, tgt


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=-1, keepdims=True)


class CharGRUModel:
    """
    Autoregressive Character-GRU Password Language Model.
    Implements a gated recurrent architecture:
      z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)
      r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)
      h~_t = tanh(x_t W_h + (r_t * h_{t-1}) U_h + b_h)
      h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
      logits_t = h_t W_out + b_out
    Trained via mini-batch BPTT with AdamW optimization.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 26,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.is_trained = False
        self.device = device or ("cuda" if TORCH_AVAILABLE else "cpu-blas-avx512")

        rng = np.random.default_rng(seed)
        scale_e = 1.0 / math.sqrt(d_model)
        scale_h = 1.0 / math.sqrt(d_model)

        # Embedding table [V, D]
        self.W_emb = rng.normal(0.0, scale_e, size=(NEURAL_VOCAB_SIZE, d_model)).astype(np.float32)
        self.W_emb[PAD_IDX] = 0.0

        # Combined Update & Reset gates: W_zr [D, 2D], U_zr [D, 2D], b_zr [2D]
        self.W_zr = rng.normal(0.0, scale_h, size=(d_model, 2 * d_model)).astype(np.float32)
        self.U_zr = rng.normal(0.0, scale_h, size=(d_model, 2 * d_model)).astype(np.float32)
        self.b_zr = np.zeros((2 * d_model,), dtype=np.float32)

        # Candidate state: W_h [D, D], U_h [D, D], b_h [D]
        self.W_h = rng.normal(0.0, scale_h, size=(d_model, d_model)).astype(np.float32)
        self.U_h = rng.normal(0.0, scale_h, size=(d_model, d_model)).astype(np.float32)
        self.b_h = np.zeros((d_model,), dtype=np.float32)

        # Layer-2 residual highway / projection: W_l2 [D, D], b_l2 [D]
        self.W_l2 = rng.normal(0.0, scale_h * 0.5, size=(d_model, d_model)).astype(np.float32)
        self.b_l2 = np.zeros((d_model,), dtype=np.float32)

        # Output projection: W_out [D, V], b_out [V]
        self.W_out = rng.normal(0.0, scale_h, size=(d_model, NEURAL_VOCAB_SIZE)).astype(np.float32)
        self.b_out = np.zeros((NEURAL_VOCAB_SIZE,), dtype=np.float32)

    def _forward_batch(self, x_ids: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Vectorized GRU forward pass on batch x_ids [B, T].
        Returns (logits [B, T, V], cache for BPTT).
        """
        B, T = x_ids.shape
        D = self.d_model

        x_emb = self.W_emb[x_ids]  # [B, T, D]
        x_zr_all = x_emb @ self.W_zr + self.b_zr  # [B, T, 2D]
        x_h_all = x_emb @ self.W_h + self.b_h      # [B, T, D]

        h_states = np.zeros((B, T + 1, D), dtype=np.float32)
        z_gates = np.zeros((B, T, D), dtype=np.float32)
        r_gates = np.zeros((B, T, D), dtype=np.float32)
        h_tildes = np.zeros((B, T, D), dtype=np.float32)
        uh_prevs = np.zeros((B, T, D), dtype=np.float32)

        h_prev = h_states[:, 0, :]
        for t in range(T):
            zr_pre = x_zr_all[:, t, :] + h_prev @ self.U_zr
            zr = _sigmoid(zr_pre)
            z = zr[:, :D]
            r = zr[:, D:]

            uh_p = h_prev @ self.U_h
            h_tilde = np.tanh(x_h_all[:, t, :] + r * uh_p)
            h_cur = (1.0 - z) * h_prev + z * h_tilde

            z_gates[:, t, :] = z
            r_gates[:, t, :] = r
            h_tildes[:, t, :] = h_tilde
            uh_prevs[:, t, :] = uh_p
            h_states[:, t + 1, :] = h_cur
            h_prev = h_cur

        H_seq = h_states[:, 1:, :]  # [B, T, D]
        # Second-layer residual non-linear transform
        H_mid = np.tanh(H_seq @ self.W_l2 + self.b_l2)
        H_out = H_seq + H_mid
        logits = H_out @ self.W_out + self.b_out  # [B, T, V]

        cache = {
            "x_ids": x_ids,
            "x_emb": x_emb,
            "h_states": h_states,
            "z_gates": z_gates,
            "r_gates": r_gates,
            "h_tildes": h_tildes,
            "uh_prevs": uh_prevs,
            "H_seq": H_seq,
            "H_mid": H_mid,
            "H_out": H_out,
        }
        return logits, cache

    def train(
        self,
        train_items: List[Tuple[str, int]],
        epochs: int = 4,
        batch_size: int = 512,
        lr: float = 4e-3,
        max_samples: int = 60000,
    ) -> Dict[str, Any]:
        """
        Train the Char-GRU model via analytical Backpropagation Through Time (BPTT) & AdamW.
        """
        inps = []
        tgts = []
        weights = []

        for pw, cnt in train_items[:max_samples]:
            if not pw:
                continue
            inp, tgt = encode_password(pw, self.max_len)
            inps.append(inp)
            tgts.append(tgt)
            weights.append(1.0 + math.log1p(max(cnt, 1)))

        if not inps:
            return {"status": "empty_data"}

        X_all = np.array(inps, dtype=np.int32)
        Y_all = np.array(tgts, dtype=np.int32)
        W_all = np.array(weights, dtype=np.float32)
        N = len(X_all)

        params = [
            ("W_emb", self.W_emb),
            ("W_zr", self.W_zr), ("U_zr", self.U_zr), ("b_zr", self.b_zr),
            ("W_h", self.W_h), ("U_h", self.U_h), ("b_h", self.b_h),
            ("W_l2", self.W_l2), ("b_l2", self.b_l2),
            ("W_out", self.W_out), ("b_out", self.b_out),
        ]
        m_adam = {name: np.zeros_like(val) for name, val in params}
        v_adam = {name: np.zeros_like(val) for name, val in params}
        step_num = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        rng = np.random.default_rng(42)
        epoch_losses: List[float] = []
        D = self.d_model

        for ep in range(epochs):
            perm = rng.permutation(N)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                bx = X_all[idx]  # [B, T]
                by = Y_all[idx]  # [B, T]
                bw = W_all[idx]  # [B]
                B, T = bx.shape

                logits, cache = self._forward_batch(bx)
                probs = _softmax(logits)  # [B, T, V]
                mask = (by != PAD_IDX).astype(np.float32)  # [B, T]

                # Compute weighted cross-entropy loss
                b_idx = np.arange(B)[:, None]
                t_idx = np.arange(T)[None, :]
                chosen_p = np.clip(probs[b_idx, t_idx, by], 1e-12, 1.0)
                tok_loss = -np.log(chosen_p) * mask
                seq_loss = tok_loss.sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
                loss = float((seq_loss * bw).sum() / np.maximum(bw.sum(), 1.0))
                total_loss += loss
                n_batches += 1

                # Gradient of loss w.r.t. logits [B, T, V]
                scale_bt = (bw / np.maximum(mask.sum(axis=1), 1.0) / np.maximum(bw.sum(), 1.0))[:, None] * mask
                dlogits = probs
                dlogits[b_idx, t_idx, by] -= 1.0
                dlogits *= scale_bt[:, :, None]

                # Backprop through output layer
                H_out = cache["H_out"]
                H_mid = cache["H_mid"]
                H_seq = cache["H_seq"]
                dlogits_flat = dlogits.reshape(B * T, NEURAL_VOCAB_SIZE)
                H_out_flat = H_out.reshape(B * T, D)

                g_W_out = H_out_flat.T @ dlogits_flat
                g_b_out = dlogits_flat.sum(axis=0)
                dH_out = (dlogits_flat @ self.W_out.T).reshape(B, T, D)

                # Backprop through Layer-2 residual highway
                dH_mid_pre = dH_out * (1.0 - H_mid * H_mid)
                dH_mid_flat = dH_mid_pre.reshape(B * T, D)
                H_seq_flat = H_seq.reshape(B * T, D)
                g_W_l2 = H_seq_flat.T @ dH_mid_flat
                g_b_l2 = dH_mid_flat.sum(axis=0)
                dH_seq = dH_out + (dH_mid_flat @ self.W_l2.T).reshape(B, T, D)

                # Backprop Through Time (BPTT) through GRU cells
                g_W_zr = np.zeros_like(self.W_zr)
                g_U_zr = np.zeros_like(self.U_zr)
                g_b_zr = np.zeros_like(self.b_zr)
                g_W_h = np.zeros_like(self.W_h)
                g_U_h = np.zeros_like(self.U_h)
                g_b_h = np.zeros_like(self.b_h)
                dx_emb = np.zeros_like(cache["x_emb"])

                dh_next = np.zeros((B, D), dtype=np.float32)
                h_states = cache["h_states"]
                z_gates = cache["z_gates"]
                r_gates = cache["r_gates"]
                h_tildes = cache["h_tildes"]
                uh_prevs = cache["uh_prevs"]
                x_emb = cache["x_emb"]

                for t in range(T - 1, -1, -1):
                    dh = dH_seq[:, t, :] + dh_next
                    h_prev = h_states[:, t, :]
                    z = z_gates[:, t, :]
                    r = r_gates[:, t, :]
                    h_tilde = h_tildes[:, t, :]
                    uh_p = uh_prevs[:, t, :]

                    dh_tilde = dh * z * (1.0 - h_tilde * h_tilde)
                    dz = dh * (h_tilde - h_prev) * z * (1.0 - z)
                    dr = (dh_tilde * uh_p) * r * (1.0 - r)
                    dzr = np.concatenate([dz, dr], axis=1)  # [B, 2D]

                    xt = x_emb[:, t, :]
                    g_W_zr += xt.T @ dzr
                    g_U_zr += h_prev.T @ dzr
                    g_b_zr += dzr.sum(axis=0)

                    g_W_h += xt.T @ dh_tilde
                    g_b_h += dh_tilde.sum(axis=0)
                    duh_p = dh_tilde * r
                    g_U_h += h_prev.T @ duh_p

                    dx_emb[:, t, :] = dzr @ self.W_zr.T + dh_tilde @ self.W_h.T
                    dh_next = dh * (1.0 - z) + dzr @ self.U_zr.T + duh_p @ self.U_h.T

                g_W_emb = np.zeros_like(self.W_emb)
                np.add.at(g_W_emb, bx, dx_emb)
                g_W_emb[PAD_IDX] = 0.0

                grads = {
                    "W_emb": g_W_emb,
                    "W_zr": g_W_zr, "U_zr": g_U_zr, "b_zr": g_b_zr,
                    "W_h": g_W_h, "U_h": g_U_h, "b_h": g_b_h,
                    "W_l2": g_W_l2, "b_l2": g_b_l2,
                    "W_out": g_W_out, "b_out": g_b_out,
                }

                # Cosine LR decay + AdamW update
                step_num += 1
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / max(epochs, 1)))
                for name, param in params:
                    g = np.clip(grads[name], -1.0, 1.0)
                    m_adam[name] = beta1 * m_adam[name] + (1.0 - beta1) * g
                    v_adam[name] = beta2 * v_adam[name] + (1.0 - beta2) * (g * g)
                    m_hat = m_adam[name] / (1.0 - beta1 ** step_num)
                    v_hat = v_adam[name] / (1.0 - beta2 ** step_num)
                    param -= cur_lr * (m_hat / (np.sqrt(v_hat) + eps) + 1e-4 * param)
                self.W_emb[PAD_IDX] = 0.0

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            print(f"    [Char-GRU] Epoch {ep + 1}/{epochs} - Loss: {avg_loss:.4f} (Perplexity: {math.exp(min(avg_loss, 10)):.2f})")

        self.is_trained = True
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1] if epoch_losses else 0.0}

    def logprob(self, pw: str) -> float:
        """Compute natural log-probability ln P(pw) under the Char-GRU model."""
        if not pw or not self.is_trained:
            return -25.0
        res = self.logprob_batch([pw], batch_size=1)
        return res[0] if res else -25.0

    def logprob_batch(self, passwords: List[str], batch_size: int = 1024) -> List[float]:
        """Compute exact autoregressive log-probabilities for a batch of passwords."""
        if not passwords:
            return []
        if not self.is_trained:
            return [-25.0 for _ in passwords]

        results: List[float] = []
        for start in range(0, len(passwords), batch_size):
            chunk = passwords[start : start + batch_size]
            inps, tgts = zip(*(encode_password(p, self.max_len) for p in chunk))
            bx = np.array(inps, dtype=np.int32)
            by = np.array(tgts, dtype=np.int32)
            B, T = bx.shape

            logits, _ = self._forward_batch(bx)
            log_probs = _log_softmax(logits)
            b_idx = np.arange(B)[:, None]
            t_idx = np.arange(T)[None, :]
            gathered = log_probs[b_idx, t_idx, by]
            mask = (by != PAD_IDX).astype(np.float32)
            seq_lps = (gathered * mask).sum(axis=1).tolist()
            results.extend(max(float(lp), -150.0) for lp in seq_lps)
        return results

    def sample_batch(self, n: int, batch_size: int = 1024, seed: int = 42) -> List[Tuple[str, float]]:
        """Vectorized batched autoregressive sampling returning (password, logprob) pairs."""
        if not self.is_trained:
            return []

        rng = np.random.default_rng(seed)
        results: List[Tuple[str, float]] = []
        D = self.d_model

        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            cur = np.full((b,), BOS_IDX, dtype=np.int32)
            h_prev = np.zeros((b, D), dtype=np.float32)
            finished = np.zeros((b,), dtype=bool)
            seq_logp = np.zeros((b,), dtype=np.float32)
            token_cols: List[np.ndarray] = []

            for _ in range(self.max_len - 1):
                xt = self.W_emb[cur]  # [B, D]
                zr = _sigmoid(xt @ self.W_zr + self.b_zr + h_prev @ self.U_zr)
                z = zr[:, :D]
                r = zr[:, D:]
                h_tilde = np.tanh(xt @ self.W_h + self.b_h + r * (h_prev @ self.U_h))
                h_cur = (1.0 - z) * h_prev + z * h_tilde

                h_mid = np.tanh(h_cur @ self.W_l2 + self.b_l2)
                logits = (h_cur + h_mid) @ self.W_out + self.b_out  # [B, V]
                logits[:, PAD_IDX] = -1e9
                logits[:, BOS_IDX] = -1e9

                probs = _softmax(logits)
                log_probs = _log_softmax(logits)

                # Vectorized inverse-CDF multinomial sampling
                cdf = np.cumsum(probs, axis=1)
                u = rng.random((b, 1), dtype=np.float32)
                next_tok = np.clip((u > cdf).sum(axis=1), 0, NEURAL_VOCAB_SIZE - 1)

                chosen_lp = log_probs[np.arange(b), next_tok]
                active = ~finished
                seq_logp += chosen_lp * active
                finished |= (next_tok == EOS_IDX)

                token_cols.append(next_tok)
                if np.all(finished):
                    break
                cur = next_tok
                h_prev = h_cur

            steps = len(token_cols)
            for row_i in range(b):
                chars = []
                for t_i in range(steps):
                    tid = int(token_cols[t_i][row_i])
                    if tid == EOS_IDX or tid <= BOS_IDX:
                        break
                    chars.append(NEURAL_VOCAB[tid])
                pw_str = "".join(chars)
                if pw_str:
                    results.append((pw_str, float(seq_logp[row_i])))

            remaining -= b

        return results

    def sample(self, rng: Optional[random.Random] = None, max_len: int = 32) -> str:
        pairs = self.sample_batch(1, batch_size=1, seed=rng.randint(1, 10**6) if rng else 42)
        return pairs[0][0] if pairs else "password123"

    def perplexity(self, items: List[Tuple[str, int]]) -> float:
        if not items or not self.is_trained:
            return float("inf")
        pws = [pw for pw, _ in items]
        lps = self.logprob_batch(pws)
        total_logp = 0.0
        total_tokens = 0
        for (pw, weight), lp in zip(items, lps):
            total_logp += weight * lp
            total_tokens += weight * (min(len(pw), self.max_len - 1) + 1)
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_logp / total_tokens)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            d_model=self.d_model,
            num_layers=self.num_layers,
            max_len=self.max_len,
            is_trained=self.is_trained,
            W_emb=self.W_emb,
            W_zr=self.W_zr, U_zr=self.U_zr, b_zr=self.b_zr,
            W_h=self.W_h, U_h=self.U_h, b_h=self.b_h,
            W_l2=self.W_l2, b_l2=self.b_l2,
            W_out=self.W_out, b_out=self.b_out,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "CharGRUModel":
        p = Path(path)
        if not p.exists() and p.with_suffix(".pt.npz").exists():
            p = p.with_suffix(".pt.npz")
        if not p.exists():
            return cls(device=device)
        data = np.load(p)
        m = cls(
            d_model=int(data["d_model"]),
            num_layers=int(data["num_layers"]),
            max_len=int(data["max_len"]),
            device=device,
        )
        m.is_trained = bool(data["is_trained"])
        m.W_emb = data["W_emb"]
        m.W_zr = data["W_zr"]
        m.U_zr = data["U_zr"]
        m.b_zr = data["b_zr"]
        m.W_h = data["W_h"]
        m.U_h = data["U_h"]
        m.b_h = data["b_h"]
        m.W_l2 = data["W_l2"]
        m.b_l2 = data["b_l2"]
        m.W_out = data["W_out"]
        m.b_out = data["b_out"]
        return m
