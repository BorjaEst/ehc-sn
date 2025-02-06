"""Test the equations module."""

import numpy as np
import pytest
from ehc_sn import CognitiveMap, equations
from numpy.testing import assert_allclose

# pylint: disable=non-ascii-name


ξ1_0 = np.array([0.5, 0.0, 0.0])  # Observation at i=0
ξ1_1 = np.array([0.0, 0.9, 0.0])  # Observation at i=1
ξ1_2 = np.array([0.0, 0.0, 0.5])  # Observation at i=2
ξ_ns = np.array([0.7, 0.5, 0.2])  # Noisy observation
Ξ1 = np.stack([ξ1_0, ξ1_1, ξ1_2])  # Observations t=1

x0 = np.array([0.9, 0.5, 0.1])  # Item at t=0
x1 = np.array([0.5, 0.9, 0.5])  # Item at t=1
x2 = np.array([0.1, 0.5, 0.9])  # Item at t=2
X1 = np.stack([x0, x1, x2])  # Items episode 1

Y1 = np.array([0.331, 0.815, 1.059])  # Sequence 1, δ = 0.3
θ1 = CognitiveMap([0.5, 0.5, 0.5])  # Cognitive map 1
Θ1 = {θ1: 0.5}  # Mixing probabilities


@pytest.mark.parametrize(
    "Ξ, desired",
    [(Ξ1, x1)],
)
def test_equation_01(Ξ, desired):
    """Test the observation code for item."""
    result = equations.get_item(Ξ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "X, desired",
    [(X1, Y1)],
)
def test_equation_02(X, desired):
    """Test the hidden code for sequence."""
    result = equations.get_sequence(X, 0.3)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ,  desired",
    [(Y1, Θ1, 0.1084)],
)
def test_equation_03(y, Θ, desired):
    """Test the probability of a sequence."""
    result = equations.p_sequence(y, Θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, θ, desired",
    [(Y1, θ1, 0.2169)],
)
def test_equation_04(y, θ, desired):
    """Test the probability of a sequence in a map."""
    result = equations.p(y, θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, θ, desired",
    [(Y1, θ1, -1.5284)],
)
def test_equation_05(y, θ, desired):
    """Test the log probability of a sequence in a map."""
    result = equations.lnp(y, θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ, desired",
    [(Y1, Θ1, 0.460)],
)
def test_equation_06(y, Θ, desired):
    """Test the mixing probabilities."""
    result = equations.z(Θ, y, τ=0.9)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ, desired",
    [(Y1, Θ1, -0.777)],
)
def test_equation_07(y, Θ, desired):
    """Test the mixing probabilities."""
    result = equations.lnz(Θ, y, τ=0.9)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "ξ, y, θ, desired",
    [(ξ_ns, Y1, θ1, [0.019, -0.565, -0.959])],
)
def test_equation_08(ξ, y, θ, desired):
    """Test the hidden code for item."""
    result = equations.item(ξ, y, θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "x, desired",
    [(x1, [0.0, 0.9, 0.0]), (x2, [0.0, 0.0, 0.9])],
)
def test_equation_09(x, desired):
    """Test the observation code for item."""
    result = equations.observation(x)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "ξ, y, θ, desired",
    [(ξ_ns, Y1, θ1, [1.048, 1.283, 1.203])],
)
def test_equation_10(ξ, y, θ, desired):
    """Test the predicted observation code."""
    result = equations.sequence(ξ, y, θ, δ=0.9)
    assert_allclose(result, desired, 1e-3)
