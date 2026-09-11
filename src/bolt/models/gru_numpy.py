"""GRU implemented from scratch in NumPy (BOLT_SPEC.md Section 6).

Same discipline as the LSTM: no framework, full BPTT, gradient clipping, Adam,
deterministic seeding, and a numerical gradient check that gates the module.

The GRU is included because the base paper never compared against one. With rare
events and limited data a smaller gated architecture frequently generalises
better than an LSTM, and that is a claim worth testing rather than assuming.

    r_t = sigmoid(W_r [x_t, h_{t-1}] + b_r)          reset
    z_t = sigmoid(W_z [x_t, h_{t-1}] + b_z)          update
    n_t = tanh   (W_n [x_t, r_t * h_{t-1}] + b_n)    candidate
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
"""

from __future__ import annotations

import numpy as np

from bolt.logging_setup import get_logger
from bolt.models.base import sample_weights_from_classes, sigmoid, weighted_bce

log = get_logger(__name__)

GATE_NAMES = ("reset", "update", "candidate")


class NumpyGRU:
    """A single-layer GRU classifier trained with class-weighted BCE."""

    name = "gru"

    def __init__(
        self,
        hidden: int = 32,
        epochs: int = 60,
        lr: float = 0.003,
        seed: int = 42,
        dropout: float = 0.1,
        clip_norm: float = 5.0,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> None:
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.dropout = float(dropout)
        self.clip_norm = float(clip_norm)
        self.batch_size = int(batch_size)
        self.verbose = verbose
        self.params: dict[str, np.ndarray] = {}
        self.history: list[float] = []
        self._adam: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._step = 0

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        h, d = self.hidden, n_features
        scale = np.sqrt(1.0 / (d + h))
        self.params = {
            # Reset and update share one matrix: rows [r | z].
            "W_rz": rng.uniform(-scale, scale, (2 * h, d + h)),
            "b_rz": np.zeros(2 * h),
            # The candidate needs its own, because its h input is gated by r.
            "W_n": rng.uniform(-scale, scale, (h, d + h)),
            "b_n": np.zeros(h),
            "W_out": rng.uniform(-scale, scale, (h, 1)),
            "b_out": np.zeros(1),
        }
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params.items()}
        self._step = 0

    def _forward(self, X: np.ndarray, training: bool = False, rng: np.random.Generator | None = None):
        n, T, d = X.shape
        h_size = self.hidden
        W_rz, b_rz = self.params["W_rz"], self.params["b_rz"]
        W_n, b_n = self.params["W_n"], self.params["b_n"]

        h = np.zeros((n, h_size))
        cache = {"X": X, "h": [h.copy()], "r": [], "z": [], "n": [], "zr": [], "zn": []}

        mask = None
        if training and self.dropout > 0.0 and rng is not None:
            keep = 1.0 - self.dropout
            mask = (rng.random((n, h_size)) < keep) / keep

        for t in range(T):
            zr = np.concatenate([X[:, t, :], h], axis=1)
            gates = zr @ W_rz.T + b_rz
            r = sigmoid(gates[:, :h_size])
            z = sigmoid(gates[:, h_size:])

            zn = np.concatenate([X[:, t, :], r * h], axis=1)
            n_t = np.tanh(zn @ W_n.T + b_n)

            h = (1.0 - z) * n_t + z * h

            cache["r"].append(r); cache["z"].append(z); cache["n"].append(n_t)
            cache["zr"].append(zr); cache["zn"].append(zn)
            cache["h"].append(h.copy())

        h_final = h
        if mask is not None:
            h_final = h_final * mask
            cache["mask"] = mask
        logits = (h_final @ self.params["W_out"] + self.params["b_out"]).ravel()
        cache["h_final"] = h_final
        return logits, cache

    def _backward(self, cache: dict, y: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
        X = cache["X"]
        n, T, d = X.shape
        h_size = self.hidden
        W_rz, W_n = self.params["W_rz"], self.params["W_n"]

        probs = sigmoid((cache["h_final"] @ self.params["W_out"] + self.params["b_out"]).ravel())
        norm = np.sum(weights)
        d_logits = (weights * (probs - y) / norm).reshape(-1, 1)

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_out"] = cache["h_final"].T @ d_logits
        grads["b_out"] = d_logits.sum(axis=0)

        dh = d_logits @ self.params["W_out"].T
        if "mask" in cache:
            dh = dh * cache["mask"]

        for t in reversed(range(T)):
            r, z, n_t = cache["r"][t], cache["z"][t], cache["n"][t]
            h_prev = cache["h"][t]
            zr, zn = cache["zr"][t], cache["zn"][t]

            dz = dh * (h_prev - n_t)
            dn = dh * (1.0 - z)
            dh_prev = dh * z

            dn_raw = dn * (1.0 - n_t ** 2)
            grads["W_n"] += dn_raw.T @ zn
            grads["b_n"] += dn_raw.sum(axis=0)

            dzn = dn_raw @ W_n
            dx_from_n = dzn[:, :d]
            d_rh = dzn[:, d:]
            dr = d_rh * h_prev
            dh_prev = dh_prev + d_rh * r

            dr_raw = dr * r * (1.0 - r)
            dz_raw = dz * z * (1.0 - z)
            d_gates = np.concatenate([dr_raw, dz_raw], axis=1)
            grads["W_rz"] += d_gates.T @ zr
            grads["b_rz"] += d_gates.sum(axis=0)

            dzr = d_gates @ W_rz
            dh = dh_prev + dzr[:, d:]

        return grads

    def _clip(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total > self.clip_norm and total > 0.0:
            factor = self.clip_norm / total
            return {k: g * factor for k, g in grads.items()}
        return grads

    def _adam_step(self, grads, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._step += 1
        for key, grad in grads.items():
            m, v = self._adam[key]
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            self._adam[key] = (m, v)
            m_hat = m / (1.0 - beta1 ** self._step)
            v_hat = v / (1.0 - beta2 ** self._step)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"NumpyGRU expects (N, T, F), got {X.shape}")

        self._init_params(X.shape[2])
        weights = (
            np.asarray(sample_weight, dtype=np.float64)
            if sample_weight is not None else sample_weights_from_classes(y)
        )
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.history = []

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_loss, batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits, cache = self._forward(X[idx], training=True, rng=rng)
                grads = self._clip(self._backward(cache, y[idx], weights[idx]))
                self._adam_step(grads)
                epoch_loss += weighted_bce(y[idx], sigmoid(logits), weights[idx])
                batches += 1
            self.history.append(epoch_loss / max(batches, 1))
            if self.verbose and (epoch + 1) % 10 == 0:
                log.info("gru epoch %d/%d loss=%.4f", epoch + 1, self.epochs, self.history[-1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.params:
            raise RuntimeError("NumpyGRU.predict_proba called before fit()")
        logits, _ = self._forward(np.asarray(X, dtype=np.float64), training=False)
        return sigmoid(logits)

    def gate_activations(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-timestep reset, update and candidate values, shape ``(N, T, hidden)``."""
        _, cache = self._forward(np.asarray(X, dtype=np.float64), training=False)
        return {
            "reset": np.stack(cache["r"], axis=1),
            "update": np.stack(cache["z"], axis=1),
            "candidate": np.stack(cache["n"], axis=1),
            "hidden": np.stack(cache["h"][1:], axis=1),
        }

    def input_gradients(self, X: np.ndarray) -> np.ndarray:
        """d(logit)/d(input) with shape ``(N, T, F)``, for gradient x input."""
        X = np.asarray(X, dtype=np.float64)
        _, cache = self._forward(X, training=False)
        n, T, d = X.shape
        W_rz, W_n = self.params["W_rz"], self.params["W_n"]

        grads_x = np.zeros_like(X)
        dh = np.tile(self.params["W_out"].T, (n, 1))

        for t in reversed(range(T)):
            r, z, n_t = cache["r"][t], cache["z"][t], cache["n"][t]
            h_prev = cache["h"][t]

            dz = dh * (h_prev - n_t)
            dn = dh * (1.0 - z)
            dh_prev = dh * z

            dn_raw = dn * (1.0 - n_t ** 2)
            dzn = dn_raw @ W_n
            dx = dzn[:, :d]
            d_rh = dzn[:, d:]
            dr = d_rh * h_prev
            dh_prev = dh_prev + d_rh * r

            d_gates = np.concatenate([dr * r * (1.0 - r), dz * z * (1.0 - z)], axis=1)
            dzr = d_gates @ W_rz
            grads_x[:, t, :] = dx + dzr[:, :d]
            dh = dh_prev + dzr[:, d:]
        return grads_x
