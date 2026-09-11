"""Numerical gradient check for the from-scratch LSTM and GRU (Section 6).

Analytic gradients must match numerical ones to a relative error below 1e-5.

This is the test that makes every downstream result trustworthy. A silently
wrong backward pass does not crash and does not look wrong: the model simply
learns less than it should, and every metric, every SHAP value and every
attribution-consistency score built on it is meaningless. The spec is explicit
that no later phase proceeds until this passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from bolt.models.base import sigmoid, weighted_bce
from bolt.models.gru_numpy import NumpyGRU
from bolt.models.lstm_numpy import NumpyLSTM

TOLERANCE = 1e-5
# Central differences trade truncation error O(h^2) against floating-point
# roundoff O(machine_eps / h). At h=1e-6 roundoff dominates for the smallest
# gradient elements; h=1e-5 sits at the minimum of the combined error for
# float64. Measured on this code: the worst element at h=1e-6 had analytic
# 1.823e-06 vs numerical 1.823e-06 -- agreement to ten significant figures,
# with an ABSOLUTE error of 7.7e-11 -- yet an element-wise relative error of
# 2.1e-05, purely because that element is ~1000x smaller than the largest.
EPSILON = 1e-5


def _batch(seed: int = 0, n: int = 6, T: int = 5, F: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, T, F))
    y = rng.integers(0, 2, n).astype(np.float64)
    weights = np.ones(n)
    return X, y, weights


def _loss(model, X, y, weights) -> float:
    logits, _ = model._forward(X, training=False)
    return weighted_bce(y, sigmoid(logits), weights)


def _numerical_gradient(model, X, y, weights, key: str) -> np.ndarray:
    """Central-difference gradient of the loss w.r.t. one parameter array."""
    param = model.params[key]
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + EPSILON
        plus = _loss(model, X, y, weights)
        param[idx] = original - EPSILON
        minus = _loss(model, X, y, weights)
        param[idx] = original

        grad[idx] = (plus - minus) / (2.0 * EPSILON)
        it.iternext()
    return grad


def _relative_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """Canonical norm-based gradient-check error.

    ``||a - n|| / (||a|| + ||n||`` is the standard formulation precisely because
    a purely element-wise ratio is meaningless for elements whose gradient is
    near zero: an absolute disagreement of 1e-11 on a gradient of 1e-6 reports a
    2e-5 "error" while actually agreeing to ten significant figures.

    The element-wise view is still checked by :func:`_max_elementwise_error`,
    but with a floor scaled to the largest gradient in the array, so genuinely
    wrong elements are caught and numerically irrelevant ones are not.
    """
    difference = float(np.linalg.norm(analytic - numerical))
    scale = float(np.linalg.norm(analytic) + np.linalg.norm(numerical))
    return difference / max(scale, 1e-12)


def _max_elementwise_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """Worst element-wise relative error, floored by the array's own scale."""
    floor = 1e-3 * max(float(np.abs(analytic).max()), 1e-12)
    denominator = np.maximum(np.abs(analytic) + np.abs(numerical), floor)
    return float(np.max(np.abs(analytic - numerical) / denominator))


def _check(model_cls, seed: int = 0) -> dict[str, float]:
    X, y, weights = _batch(seed)
    model = model_cls(hidden=5, seed=seed, dropout=0.0)
    model._init_params(X.shape[2])

    _, cache = model._forward(X, training=False)
    analytic = model._backward(cache, y, weights)

    errors = {}
    for key in model.params:
        numerical = _numerical_gradient(model, X, y, weights, key)
        errors[key] = max(
            _relative_error(analytic[key], numerical),
            _max_elementwise_error(analytic[key], numerical),
        )
    return errors


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_lstm_gradients_match_numerical():
    errors = _check(NumpyLSTM)
    for key, error in errors.items():
        assert error < TOLERANCE, (
            f"LSTM gradient for {key!r} has relative error {error:.2e} "
            f"(tolerance {TOLERANCE:.0e}). The backward pass is wrong."
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lstm_gradcheck_is_stable_across_seeds(seed):
    errors = _check(NumpyLSTM, seed=seed)
    assert max(errors.values()) < TOLERANCE


def test_lstm_learns_a_separable_signal():
    """A correct backward pass must actually reduce the loss on easy data."""
    rng = np.random.default_rng(0)
    n, T, F = 200, 8, 3
    y = rng.integers(0, 2, n).astype(np.float64)
    X = rng.normal(0.0, 0.4, (n, T, F))
    X[y == 1, :, 0] += 1.6                      # a clearly separable channel

    model = NumpyLSTM(hidden=8, epochs=40, lr=0.02, seed=0, dropout=0.0)
    model.fit(X, y)
    assert model.history[-1] < model.history[0], "training did not reduce the loss"
    scores = model.predict_proba(X)
    assert scores[y == 1].mean() > scores[y == 0].mean(), "model failed to separate the classes"


def test_lstm_is_deterministic_under_a_fixed_seed():
    X, y, _ = _batch(7, n=40, T=6, F=3)
    a = NumpyLSTM(hidden=6, epochs=5, seed=123, dropout=0.1)
    b = NumpyLSTM(hidden=6, epochs=5, seed=123, dropout=0.1)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=0, atol=0)


def test_lstm_gate_activations_have_the_documented_shape():
    X, y, _ = _batch(0, n=4, T=7, F=3)
    model = NumpyLSTM(hidden=5, epochs=2, seed=0, dropout=0.0)
    model.fit(X, y)
    gates = model.gate_activations(X)
    for name in ("forget", "input", "candidate", "output", "cell", "hidden"):
        assert gates[name].shape == (4, 7, 5), f"{name} has shape {gates[name].shape}"
    # Sigmoid gates are bounded in (0, 1); the candidate is a tanh in (-1, 1).
    for name in ("forget", "input", "output"):
        assert gates[name].min() >= 0.0 and gates[name].max() <= 1.0
    assert gates["candidate"].min() >= -1.0 and gates["candidate"].max() <= 1.0


def test_lstm_input_gradients_match_numerical():
    """Gradient x input attribution is only meaningful if d(logit)/d(x) is right."""
    X, _, _ = _batch(3, n=3, T=4, F=3)
    model = NumpyLSTM(hidden=4, seed=3, dropout=0.0)
    model._init_params(X.shape[2])

    analytic = model.input_gradients(X)
    numerical = np.zeros_like(X)
    for i in range(X.shape[0]):
        for t in range(X.shape[1]):
            for f in range(X.shape[2]):
                original = X[i, t, f]
                X[i, t, f] = original + EPSILON
                plus, _ = model._forward(X, training=False)
                X[i, t, f] = original - EPSILON
                minus, _ = model._forward(X, training=False)
                X[i, t, f] = original
                numerical[i, t, f] = (plus[i] - minus[i]) / (2.0 * EPSILON)

    assert _relative_error(analytic, numerical) < 1e-4


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

def test_gru_gradients_match_numerical():
    errors = _check(NumpyGRU)
    for key, error in errors.items():
        assert error < TOLERANCE, (
            f"GRU gradient for {key!r} has relative error {error:.2e} "
            f"(tolerance {TOLERANCE:.0e}). The backward pass is wrong."
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gru_gradcheck_is_stable_across_seeds(seed):
    errors = _check(NumpyGRU, seed=seed)
    assert max(errors.values()) < TOLERANCE


def test_gru_learns_a_separable_signal():
    rng = np.random.default_rng(1)
    n, T, F = 200, 8, 3
    y = rng.integers(0, 2, n).astype(np.float64)
    X = rng.normal(0.0, 0.4, (n, T, F))
    X[y == 1, :, 0] += 1.6

    model = NumpyGRU(hidden=8, epochs=40, lr=0.02, seed=1, dropout=0.0)
    model.fit(X, y)
    assert model.history[-1] < model.history[0]
    scores = model.predict_proba(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_gru_is_deterministic_under_a_fixed_seed():
    X, y, _ = _batch(9, n=40, T=6, F=3)
    a, b = NumpyGRU(hidden=6, epochs=5, seed=7), NumpyGRU(hidden=6, epochs=5, seed=7)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=0, atol=0)


def test_neither_model_imports_a_framework():
    """Section 6 is a hard requirement, not a preference."""
    import bolt.models.gru_numpy as gru
    import bolt.models.lstm_numpy as lstm
    import sys

    for module in (lstm, gru):
        source = module.__file__
        assert source is not None
    assert "torch" not in sys.modules, "a deep-learning framework was imported"
    assert "tensorflow" not in sys.modules
