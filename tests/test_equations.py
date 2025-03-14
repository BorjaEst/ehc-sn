"""Test the equations module."""

import numpy as np
import pytest
from ehc_sn import equations
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

Y1 = np.array([0.0, 0.5, 1.0])  # Sequence 1, δ = 0.3
v1 = np.array([0.0, 0.5, 0.5])  # Velocity 1
θ1 = np.array([0.0, 0.5, 0.5])  # Cognitive map 1
θ2 = np.array([0.2, 0.0, 0.8])  # Cognitive map 2
Θ1 = [θ1, θ2]  # Cognitive maps 1

z1 = np.array([0.5, 0.5])  # Mixing probabilities 1
π1 = np.array([0.5, 0.1])  # Mixing hyperparameters 1
ρ1 = np.array([[0.5, 0.5, 0.5]] * 2)  # Mixing hyperparameters 1


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
    [(X1, [0.331, 0.815, 1.059])],
)
def test_equation_02(X, desired):
    """Test the hidden code for sequence."""
    result = equations.get_sequence(X, 0.3)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ, z, desired",
    [(Y1, Θ1, z1, 0.1768)],
)
def test_equation_03(y, Θ, z, desired):
    """Test the probability of a sequence."""
    result = equations.p_sequence(y, Θ, z)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, θ, desired",
    [(Y1, θ1, 0.3536), (Y1, θ2, 0.0)],
)
def test_equation_04(y, θ, desired):
    """Test the probability of a sequence in a map."""
    result = equations.p(y, θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, θ, desired",
    [(Y1, θ1, -1.040), (Y1, θ2, -np.inf)],
)
def test_equation_05(y, θ, desired):
    """Test the log probability of a sequence in a map."""
    result = equations.lnp(y, θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ, z, desired",
    [(Y1, Θ1, z1, [0.483, 0.0])],
)
def test_equation_06(y, Θ, z, desired):
    """Test the mixing probabilities."""
    result = equations.mixing(y, Θ, z, τ=0.9)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "y, Θ, z, desired",
    [(Y1, Θ1, z1, [-0.728, -np.inf])],
)
def test_equation_07(y, Θ, z, desired):
    """Test the mixing probabilities."""
    result = equations.lnz(y, Θ, z, τ=0.9)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "ξ, y, θ, v, desired",
    [(ξ_ns, Y1, θ1, v1, [0.0, -0.25, -0.9])],
)
def test_equation_08(ξ, y, θ, v, desired):
    """Test the hidden code for item."""
    results = [equations.item(ξ, y, θ, v, c=1e-3) for _ in range(99)]
    assert_allclose(np.mean(results, axis=0), desired, 0.2)


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
    [(ξ_ns, Y1, θ1, [0.7, 1.0, 1.15])],
)
def test_equation_10(ξ, y, θ, desired):
    """Test the predicted observation code."""
    result = equations.sequence(ξ, y, θ, δ=0.9)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "π, z, k, desired",
    [(π1, z1, 0, 0.95)],
)
def test_equation_11(π, z, k, desired):
    """Test the mixing hyperparameters."""
    result = equations.π_update(π[k], z[k], γ=0.1)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "ρ, z, y, k, desired",
    [(ρ1, z1, Y1, 0, [0.50, 0.75, 1.00])],
)
def test_equation_12(ρ, z, y, k, desired):
    """Test the mixing hyperparameters."""
    result = equations.ρ_update(ρ[k], z[k], y)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize(
    "Θ, ξ, x, k, desired",
    [(Θ1, ξ1_1 + ξ1_2, x1, 0, [0.0, 0.555, 1.143])],
)
def test_equation_13(Θ, ξ, x, k, desired):
    """Test the mixing hyperparameters."""
    result = equations.pmap_update(Θ[k], ξ, x, λ=0.1)
    assert_allclose(result, desired, 1e-3)
