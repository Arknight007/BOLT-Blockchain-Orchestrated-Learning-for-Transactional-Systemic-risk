"""LSTM implemented from scratch in NumPy (BOLT_SPEC.md Section 6).

No PyTorch, no TensorFlow. The from-scratch implementation is part of the
contribution: it exposes gate activations directly, which the gradient-based
attribution in ``explain/attribution.py`` depends on, and it demonstrates command
of the architecture rather than dependence on a framework abstraction.

Standard formulation, with gates concatenated into one weight matrix for speed:

    f_t = sigmoid(W_f [x_t, h_{t-1}] + b_f)     forget
    i_t = sigmoid(W_i [x_t, h_{t-1}] + b_i)     input
    g_t = tanh   (W_g [x_t, h_{t-1}] + b_g)     candidate
    o_t = sigmoid(W_o [x_t, h_{t-1}] + b_o)     output
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)

The prediction head reads the FINAL hidden state. Backpropagation through time
runs the full sequence with gradient clipping at the configured norm, optimised
with Adam.

Correctness is established by ``tests/test_lstm_gradcheck.py``, which compares
analytic gradients against numerical ones. A silently wrong backward pass would
invalidate every downstream result, so that test gates this module.
"""

from __future__ import annotations

import numpy as np

from bolt.logging_setup import get_logger
from bolt.models.base import sample_weights_from_classes, sigmoid, weighted_bce

log = get_logger(__name__)

GATE_NAMES = ("forget", "input", "candidate", "output")


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class NumpyLSTM:
    """A single-layer LSTM classifier trained with class-weighted BCE."""

    name = "lstm"

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

    # -- initialisation ----------------------------------------------------
    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        h, d = self.hidden, n_features
        # Xavier scaling over the concatenated [x, h] input.
        scale = np.sqrt(1.0 / (d + h))
        self.params = {
            # One matrix for all four gates: rows [f | i | g | o].
            "W": rng.uniform(-scale, scale, (4 * h, d + h)),
            "b": np.zeros(4 * h),
            "W_out": rng.uniform(-scale, scale, (h, 1)),
            "b_out": np.zeros(1),
        }
        # Forget-gate bias initialised to 1.0: the standard trick that stops the
        # cell state being wiped early in training, which matters here because a
        # crash signal builds over the whole 30-day window.
        self.params["b"][:h] = 1.0
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params.items()}
        self._step = 0

    # -- forward -----------------------------------------------------------
    def _forward(self, X: np.ndarray, training: bool = False, rng: np.random.Generator | None = None):
        """Run the sequence. Returns (logits, cache)."""
        n, T, d = X.shape
        h_size = self.hidden
        W, b = self.params["W"], self.params["b"]

        h = np.zeros((n, h_size))
        c = np.zeros((n, h_size))
        cache = {
            "X": X, "h": [h.copy()], "c": [c.copy()],
            "f": [], "i": [], "g": [], "o": [], "tanh_c": [], "z": [],
        }

        mask = None
        if training and self.dropout > 0.0 and rng is not None:
            keep = 1.0 - self.dropout
            mask = (rng.random((n, h_size)) < keep) / keep

        for t in range(T):
            z = np.concatenate([X[:, t, :], h], axis=1)          # (n, d+h)
            gates = z @ W.T + b                                   # (n, 4h)
            f = sigmoid(gates[:, :h_size])
            i = sigmoid(gates[:, h_size:2 * h_size])
            g = _tanh(gates[:, 2 * h_size:3 * h_size])
            o = sigmoid(gates[:, 3 * h_size:])

            c = f * c + i * g
            tanh_c = _tanh(c)
            h = o * tanh_c

            cache["z"].append(z)
            cache["f"].append(f); cache["i"].append(i)
            cache["g"].append(g); cache["o"].append(o)
            cache["c"].append(c.copy()); cache["h"].append(h.copy())
            cache["tanh_c"].append(tanh_c)

        h_final = h
        if mask is not None:
            h_final = h_final * mask
            cache["mask"] = mask
        logits = (h_final @ self.params["W_out"] + self.params["b_out"]).ravel()
        cache["h_final"] = h_final
        return logits, cache

    # -- backward ----------------------------------------------------------
    def _backward(self, cache: dict, y: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation through time. Returns gradients keyed like params."""
        X = cache["X"]
        n, T, _ = X.shape
        h_size = self.hidden
        W = self.params["W"]

        probs = sigmoid((cache["h_final"] @ self.params["W_out"] + self.params["b_out"]).ravel())
        # d(weighted BCE)/d(logit) for a sigmoid output collapses to (p - y).
        norm = np.sum(weights)
        d_logits = (weights * (probs - y) / norm).reshape(-1, 1)      # (n, 1)

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_out"] = cache["h_final"].T @ d_logits
        grads["b_out"] = d_logits.sum(axis=0)

        dh = d_logits @ self.params["W_out"].T                        # (n, h)
        if "mask" in cache:
            dh = dh * cache["mask"]
        dc = np.zeros((n, h_size))

        for t in reversed(range(T)):
            f, i, g, o = cache["f"][t], cache["i"][t], cache["g"][t], cache["o"][t]
            tanh_c = cache["tanh_c"][t]
            c_prev = cache["c"][t]          # cache["c"][0] is the initial state
            z = cache["z"][t]

            do = dh * tanh_c
            dc = dc + dh * o * (1.0 - tanh_c ** 2)

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            # Through the activations.
            df_raw = df * f * (1.0 - f)
            di_raw = di * i * (1.0 - i)
            dg_raw = dg * (1.0 - g ** 2)
            do_raw = do * o * (1.0 - o)

            d_gates = np.concatenate([df_raw, di_raw, dg_raw, do_raw], axis=1)   # (n, 4h)
            grads["W"] += d_gates.T @ z
            grads["b"] += d_gates.sum(axis=0)

            dz = d_gates @ W                                           # (n, d+h)
            dh = dz[:, X.shape[2]:]
            dc = dc_prev

        return grads

    # -- optimisation ------------------------------------------------------
    def _clip(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total > self.clip_norm and total > 0.0:
            factor = self.clip_norm / total
            return {k: g * factor for k, g in grads.items()}
        return grads

    def _adam_step(self, grads: dict[str, np.ndarray],
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._step += 1
        for key, grad in grads.items():
            m, v = self._adam[key]
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            self._adam[key] = (m, v)
            m_hat = m / (1.0 - beta1 ** self._step)
            v_hat = v / (1.0 - beta2 ** self._step)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # -- public API --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"NumpyLSTM expects (N, T, F), got {X.shape}")

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
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits, cache = self._forward(X[idx], training=True, rng=rng)
                grads = self._clip(self._backward(cache, y[idx], weights[idx]))
                self._adam_step(grads)
                epoch_loss += weighted_bce(y[idx], sigmoid(logits), weights[idx])
                batches += 1
            self.history.append(epoch_loss / max(batches, 1))
            if self.verbose and (epoch + 1) % 10 == 0:
                log.info("lstm epoch %d/%d loss=%.4f", epoch + 1, self.epochs, self.history[-1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.params:
            raise RuntimeError("NumpyLSTM.predict_proba called before fit()")
        logits, _ = self._forward(np.asarray(X, dtype=np.float64), training=False)
        return sigmoid(logits)

    def gate_activations(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-timestep gate values, for the attribution module.

        Returns arrays of shape ``(N, T, hidden)`` keyed by :data:`GATE_NAMES`,
        plus the cell and hidden states. This is what a framework abstraction
        would hide and the report claims direct access to.
        """
        _, cache = self._forward(np.asarray(X, dtype=np.float64), training=False)
        return {
            "forget": np.stack(cache["f"], axis=1),
            "input": np.stack(cache["i"], axis=1),
            "candidate": np.stack(cache["g"], axis=1),
            "output": np.stack(cache["o"], axis=1),
            "cell": np.stack(cache["c"][1:], axis=1),
            "hidden": np.stack(cache["h"][1:], axis=1),
        }

    def input_gradients(self, X: np.ndarray) -> np.ndarray:
        """d(logit)/d(input) with shape ``(N, T, F)``, for gradient x input."""
        X = np.asarray(X, dtype=np.float64)
        _, cache = self._forward(X, training=False)
        n, T, d = X.shape
        h_size = self.hidden
        W = self.params["W"]

        grads_x = np.zeros_like(X)
        dh = np.tile(self.params["W_out"].T, (n, 1))
        dc = np.zeros((n, h_size))

        for t in reversed(range(T)):
            f, i, g, o = cache["f"][t], cache["i"][t], cache["g"][t], cache["o"][t]
            tanh_c = cache["tanh_c"][t]
            c_prev = cache["c"][t]

            do = dh * tanh_c
            dc = dc + dh * o * (1.0 - tanh_c ** 2)
            d_gates = np.concatenate([
                dc * c_prev * f * (1.0 - f),
                dc * g * i * (1.0 - i),
                dc * i * (1.0 - g ** 2),
                do * o * (1.0 - o),
            ], axis=1)

            dz = d_gates @ W
            grads_x[:, t, :] = dz[:, :d]
            dh = dz[:, d:]
            dc = dc * f
        return grads_x
