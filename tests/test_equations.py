import numpy as np
import pytest
from ehc_sn import CognitiveMap, equations
from numpy.testing import assert_allclose

# pylint: disable=non-ascii-name


Ξ1 = np.array(  # Observation at t=1
    [
        [0.5, 0.0, 0.0],  # Observation at i=0
        [0.0, 0.9, 0.0],  # Observation at i=1
        [0.0, 0.0, 0.5],  # Observation at i=2
    ]
)

X0 = np.array(
    [
        [0.9, 0.5, 0.1],  # Item at t=0
        [0.5, 0.9, 0.5],  # Item at t=1
        [0.1, 0.5, 0.9],  # Item at t=2
    ]
)

Y0 = np.array([0.331, 0.815, 1.059])  # Trajectory δ = 0.3

θ1 = CognitiveMap(np.array([0.5, 0.5, 0.5]))  # Cognitive map 1


@pytest.mark.parametrize("Ξ, desired", [(Ξ1, X0[1])])
def test_equation_01(Ξ, desired):
    """Test the observation code for item."""
    result = equations.item(Ξ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize("X, desired", [(X0, Y0)])
def test_equation_02(X, desired):
    """Test the hidden code for trajectory."""
    result = equations.trajectory(X, 0.3)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize("y, Θ,  desired", [(Y0, {θ1: 0.5}, 0.1084)])
def test_equation_03(y, Θ, desired):
    """Test the probability of a trajectory."""
    result = equations.p_trajectory(y, Θ)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize("y, θ, desired", [(Y0, θ1, 0.2169)])
def test_equation_04(y, θ, desired):
    """Test the probability of a trajectory in a map."""
    result = equations.p(y, θ, γ=1.0)
    assert_allclose(result, desired, 1e-3)


@pytest.mark.parametrize("y, θ, desired", [(Y0, θ1, -1.5284)])
def test_equation_05(y, θ, desired):
    """Test the log probability of a trajectory in a map."""
    result = equations.lnp(y, θ, γ=1.0)
    assert_allclose(result, desired, 1e-3)
