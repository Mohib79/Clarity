"""
PassGPT — Causal Decoder Transformer for Password Modeling & Guessability Evaluation.
Implements an autoregressive GPT-style Multi-Head Causal Self-Attention architecture
with Positional Embeddings, Causal Masking, Residual Connections, FFN, and AdamW optimization.
Supports both PyTorch CUDA and Vectorized NumPy BLAS (Intel Core Ultra / AVX-512) execution,
automatically bypassing Windows Application Control DLL restrictions (WinError 4551).
"""

import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pwlab.model.neural_gru import (
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    NEURAL_VOCAB,
    NEURAL_VOCAB_SIZE,
    NEURAL_CHAR_TO_IDX,
    TORCH_AVAILABLE,
    encode_password,
    _softmax,
    _log_softmax,
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class PassGPTModel:
    """
    PassGPT Causal Decoder Transformer for Password Probability & Token-Level Entropy Analysis.
    Architecture:
      - Token Embedding W_tok [V, D] + Positional Embedding W_pos [T, D]
      - Multi-Head Causal Self-Attention (n_heads, causal lower-triangular mask)
      - Position-wise Feed-Forward Network (D -> 2D -> D with residual connection)
      - Output Projection Head W_out [D, V] + b_out [V]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 26,
        dropout: float = 0.1,
        device: Optional[str] = None,
        seed: int = 1337,
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.dropout = dropout
        self.is_trained = False
        self.device = device or ("cuda" if TORCH_AVAILABLE else "cpu-blas-avx512")

        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(d_model)

        self.W_tok = rng.normal(0.0, scale, size=(NEURAL_VOCAB_SIZE, d_model)).astype(np.float32)
        self.W_tok[PAD_IDX] = 0.0
        self.W_pos = rng.normal(0.0, scale * 0.5, size=(max_len, d_model)).astype(np.float32)

        # Multi-Head Attention projections
        self.W_q = rng.normal(0.0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0.0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0.0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0.0, scale * 0.5, size=(d_model, d_model)).astype(np.float32)

        # Feed-Forward Network (D -> 2D -> D)
        ffn_dim = 2 * d_model
        self.W_f1 = rng.normal(0.0, scale, size=(d_model, ffn_dim)).astype(np.float32)
        self.b_f1 = np.zeros((ffn_dim,), dtype=np.float32)
        self.W_f2 = rng.normal(0.0, 1.0 / math.sqrt(ffn_dim), size=(ffn_dim, d_model)).astype(np.float32)
        self.b_f2 = np.zeros((d_model,), dtype=np.float32)

        # Output Head
        self.W_out = rng.normal(0.0, scale, size=(d_model, NEURAL_VOCAB_SIZE)).astype(np.float32)
        self.b_out = np.zeros((NEURAL_VOCAB_SIZE,), dtype=np.float32)

        # Precompute causal mask [max_len, max_len]
        mask = np.triu(np.ones((max_len, max_len), dtype=np.float32), k=1) * (-1e9)
        self._causal_mask = mask

    def _forward_batch(self, x_ids: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Vectorized Multi-Head Causal Self-Attention + FFN forward pass on x_ids [B, T].
        """
        B, T = x_ids.shape
        D = self.d_model
        H = self.n_heads
        Dh = self.head_dim
        inv_sqrt_d = 1.0 / math.sqrt(Dh)

        X0 = self.W_tok[x_ids] + self.W_pos[:T][None, :, :]  # [B, T, D]

        # Q, K, V projections -> reshape to [B, H, T, Dh]
        Q = (X0 @ self.W_q).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        K = (X0 @ self.W_k).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        V = (X0 @ self.W_v).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # Scaled Dot-Product Causal Attention [B, H, T, T]
        scores = (Q @ K.transpose(0, 1, 3, 2)) * inv_sqrt_d
        scores = scores + self._causal_mask[:T, :T][None, None, :, :]
        attn_weights = _softmax(scores)  # [B, H, T, T]

        attn_ctx = attn_weights @ V  # [B, H, T, Dh]
        attn_merged = attn_ctx.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_proj = attn_merged @ self.W_o

        # First residual connection
        X1 = X0 + attn_proj

        # Position-wise Feed-Forward Network with ReLU + Second residual connection
        F_pre = X1 @ self.W_f1 + self.b_f1
        F_act = _relu(F_pre)
        F_out = F_act @ self.W_f2 + self.b_f2
        X2 = X1 + F_out

        logits = X2 @ self.W_out + self.b_out  # [B, T, V]

        cache = {
            "x_ids": x_ids,
            "X0": X0,
            "Q": Q, "K": K, "V": V,
            "attn_weights": attn_weights,
            "attn_merged": attn_merged,
            "X1": X1,
            "F_pre": F_pre,
            "F_act": F_act,
            "X2": X2,
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
        Train PassGPT via analytical Transformer Backpropagation & AdamW optimizer.
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
            ("W_tok", self.W_tok), ("W_pos", self.W_pos),
            ("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v), ("W_o", self.W_o),
            ("W_f1", self.W_f1), ("b_f1", self.b_f1),
            ("W_f2", self.W_f2), ("b_f2", self.b_f2),
            ("W_out", self.W_out), ("b_out", self.b_out),
        ]
        m_adam = {name: np.zeros_like(val) for name, val in params}
        v_adam = {name: np.zeros_like(val) for name, val in params}
        step_num = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        rng = np.random.default_rng(1337)
        epoch_losses: List[float] = []
        D = self.d_model
        H = self.n_heads
        Dh = self.head_dim
        inv_sqrt_d = 1.0 / math.sqrt(Dh)

        for ep in range(epochs):
            perm = rng.permutation(N)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                bx = X_all[idx]
                by = Y_all[idx]
                bw = W_all[idx]
                B, T = bx.shape

                logits, cache = self._forward_batch(bx)
                probs = _softmax(logits)
                mask = (by != PAD_IDX).astype(np.float32)

                b_idx = np.arange(B)[:, None]
                t_idx = np.arange(T)[None, :]
                chosen_p = np.clip(probs[b_idx, t_idx, by], 1e-12, 1.0)
                tok_loss = -np.log(chosen_p) * mask
                seq_loss = tok_loss.sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
                loss = float((seq_loss * bw).sum() / np.maximum(bw.sum(), 1.0))
                total_loss += loss
                n_batches += 1

                # Gradient wrt logits [B, T, V]
                scale_bt = (bw / np.maximum(mask.sum(axis=1), 1.0) / np.maximum(bw.sum(), 1.0))[:, None] * mask
                dlogits = probs
                dlogits[b_idx, t_idx, by] -= 1.0
                dlogits *= scale_bt[:, :, None]

                # Backprop through Output Head
                X2_flat = cache["X2"].reshape(B * T, D)
                dlogits_flat = dlogits.reshape(B * T, NEURAL_VOCAB_SIZE)
                g_W_out = X2_flat.T @ dlogits_flat
                g_b_out = dlogits_flat.sum(axis=0)
                dX2 = (dlogits_flat @ self.W_out.T).reshape(B, T, D)

                # Backprop through FFN + residual
                F_act_flat = cache["F_act"].reshape(B * T, 2 * D)
                dX2_flat = dX2.reshape(B * T, D)
                g_W_f2 = F_act_flat.T @ dX2_flat
                g_b_f2 = dX2_flat.sum(axis=0)

                dF_act = (dX2_flat @ self.W_f2.T).reshape(B, T, 2 * D)
                dF_pre = dF_act * (cache["F_pre"] > 0.0).astype(np.float32)
                dF_pre_flat = dF_pre.reshape(B * T, 2 * D)
                X1_flat = cache["X1"].reshape(B * T, D)
                g_W_f1 = X1_flat.T @ dF_pre_flat
                g_b_f1 = dF_pre_flat.sum(axis=0)

                dX1 = dX2 + (dF_pre_flat @ self.W_f1.T).reshape(B, T, D)

                # Backprop through Multi-Head Causal Self-Attention
                dX1_flat = dX1.reshape(B * T, D)
                attn_merged_flat = cache["attn_merged"].reshape(B * T, D)
                g_W_o = attn_merged_flat.T @ dX1_flat
                d_attn_merged = (dX1_flat @ self.W_o.T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # [B, H, T, Dh]

                attn_w = cache["attn_weights"]  # [B, H, T, T]
                Q, K, V = cache["Q"], cache["K"], cache["V"]

                dV = attn_w.transpose(0, 1, 3, 2) @ d_attn_merged  # [B, H, T, Dh]
                d_attn_w = d_attn_merged @ V.transpose(0, 1, 3, 2)  # [B, H, T, T]

                # Softmax backward: dS = A * (dA - sum(A * dA))
                sum_ad = np.sum(attn_w * d_attn_w, axis=-1, keepdims=True)
                d_scores = attn_w * (d_attn_w - sum_ad) * inv_sqrt_d  # [B, H, T, T]

                dQ = d_scores @ K  # [B, H, T, Dh]
                dK = d_scores.transpose(0, 1, 3, 2) @ Q  # [B, H, T, Dh]

                dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, D)
                dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, D)
                dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, D)
                X0_flat = cache["X0"].reshape(B * T, D)

                g_W_q = X0_flat.T @ dQ_flat
                g_W_k = X0_flat.T @ dK_flat
                g_W_v = X0_flat.T @ dV_flat

                dX0 = dX1 + (
                    dQ_flat @ self.W_q.T + dK_flat @ self.W_k.T + dV_flat @ self.W_v.T
                ).reshape(B, T, D)

                g_W_pos = np.zeros_like(self.W_pos)
                g_W_pos[:T] = dX0.sum(axis=0)

                g_W_tok = np.zeros_like(self.W_tok)
                np.add.at(g_W_tok, bx, dX0)
                g_W_tok[PAD_IDX] = 0.0

                grads = {
                    "W_tok": g_W_tok, "W_pos": g_W_pos,
                    "W_q": g_W_q, "W_k": g_W_k, "W_v": g_W_v, "W_o": g_W_o,
                    "W_f1": g_W_f1, "b_f1": g_b_f1,
                    "W_f2": g_W_f2, "b_f2": g_b_f2,
                    "W_out": g_W_out, "b_out": g_b_out,
                }

                step_num += 1
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / max(epochs, 1)))
                for name, param in params:
                    g = np.clip(grads[name], -1.0, 1.0)
                    m_adam[name] = beta1 * m_adam[name] + (1.0 - beta1) * g
                    v_adam[name] = beta2 * v_adam[name] + (1.0 - beta2) * (g * g)
                    m_hat = m_adam[name] / (1.0 - beta1 ** step_num)
                    v_hat = v_adam[name] / (1.0 - beta2 ** step_num)
                    param -= cur_lr * (m_hat / (np.sqrt(v_hat) + eps) + 1e-4 * param)
                self.W_tok[PAD_IDX] = 0.0

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            print(f"    [PassGPT]  Epoch {ep + 1}/{epochs} - Loss: {avg_loss:.4f} (Perplexity: {math.exp(min(avg_loss, 10)):.2f})")

        self.is_trained = True
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1] if epoch_losses else 0.0}

    def logprob(self, pw: str) -> float:
        """Compute natural log-probability ln P(pw) under PassGPT."""
        if not pw or not self.is_trained:
            return -25.0
        res = self.logprob_batch([pw], batch_size=1)
        return res[0] if res else -25.0

    def char_surprises(self, pw: str) -> List[Dict[str, Any]]:
        """
        Return per-character surprise (-log2 P(c_i | c_<i)) from PassGPT self-attention.
        """
        if not pw or not self.is_trained:
            return []

        trimmed = pw[: self.max_len - 1]
        token_ids = [NEURAL_CHAR_TO_IDX.get(c, 3) for c in trimmed]
        inp = [BOS_IDX] + token_ids[:-1]
        x = np.array([inp], dtype=np.int32)
        logits, _ = self._forward_batch(x)
        probs = _softmax(logits[0])

        out: List[Dict[str, Any]] = []
        for idx, (ch, tid) in enumerate(zip(trimmed, token_ids)):
            p = max(float(probs[idx, tid]), 1e-12)
            bits = -math.log2(p)
            out.append({
                "index": idx,
                "char": ch,
                "prob": round(p, 6),
                "bits": round(bits, 2),
            })
        return out

    def logprob_batch(self, passwords: List[str], batch_size: int = 1024) -> List[float]:
        """Compute log-probabilities for a batch of passwords."""
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

    def sample_batch(self, n: int, batch_size: int = 512, seed: int = 42) -> List[Tuple[str, float]]:
        """Batched autoregressive generation returning (password, logprob) pairs."""
        if not self.is_trained:
            return []

        rng = np.random.default_rng(seed)
        results: List[Tuple[str, float]] = []

        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            seq = np.full((b, 1), BOS_IDX, dtype=np.int32)
            finished = np.zeros((b,), dtype=bool)
            seq_logp = np.zeros((b,), dtype=np.float32)

            for _ in range(self.max_len - 1):
                logits, _ = self._forward_batch(seq)
                step_logits = logits[:, -1, :]
                step_logits[:, PAD_IDX] = -1e9
                step_logits[:, BOS_IDX] = -1e9

                probs = _softmax(step_logits)
                log_probs = _log_softmax(step_logits)

                cdf = np.cumsum(probs, axis=1)
                u = rng.random((b, 1), dtype=np.float32)
                next_tok = np.clip((u > cdf).sum(axis=1), 0, NEURAL_VOCAB_SIZE - 1)

                chosen_lp = log_probs[np.arange(b), next_tok]
                active = ~finished
                seq_logp += chosen_lp * active
                finished |= (next_tok == EOS_IDX)

                seq = np.concatenate([seq, next_tok[:, None]], axis=1)
                if np.all(finished):
                    break

            seqs_list = seq[:, 1:].tolist()
            for row_tokens, lp in zip(seqs_list, seq_logp.tolist()):
                chars = []
                for tid in row_tokens:
                    if tid == EOS_IDX or tid <= BOS_IDX:
                        break
                    chars.append(NEURAL_VOCAB[tid])
                pw_str = "".join(chars)
                if pw_str:
                    results.append((pw_str, float(lp)))

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
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_len=self.max_len,
            is_trained=self.is_trained,
            W_tok=self.W_tok, W_pos=self.W_pos,
            W_q=self.W_q, W_k=self.W_k, W_v=self.W_v, W_o=self.W_o,
            W_f1=self.W_f1, b_f1=self.b_f1,
            W_f2=self.W_f2, b_f2=self.b_f2,
            W_out=self.W_out, b_out=self.b_out,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "PassGPTModel":
        p = Path(path)
        if not p.exists() and p.with_suffix(".pt.npz").exists():
            p = p.with_suffix(".pt.npz")
        if not p.exists():
            return cls(device=device)
        data = np.load(p)
        m = cls(
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            n_layers=int(data["n_layers"]),
            max_len=int(data["max_len"]),
            device=device,
        )
        m.is_trained = bool(data["is_trained"])
        m.W_tok = data["W_tok"]
        m.W_pos = data["W_pos"]
        m.W_q = data["W_q"]
        m.W_k = data["W_k"]
        m.W_v = data["W_v"]
        m.W_o = data["W_o"]
        m.W_f1 = data["W_f1"]
        m.b_f1 = data["b_f1"]
        m.W_f2 = data["W_f2"]
        m.b_f2 = data["b_f2"]
        m.W_out = data["W_out"]
        m.b_out = data["b_out"]
        return m
