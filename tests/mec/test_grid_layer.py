import numpy as np
import pytest


@pytest.mark.parametrize("scale", [0.2])
def test_grid_cell_toroid_x_cycling(layer):
    """Test that grid cell activation cycles correctly when moving along x-axis."""
    # Get width, height and spacing
    width, height = layer.width, layer.height
    spacing = layer.spacing
    
    # Define reference point and a point one full period away on x-axis
    reference_point = (0.5, 0.5)
    cycled_point = (0.5 + width * spacing, 0.5)
    
    # Measure activations at both points
    layer.update_activity(reference_point)
    reference_activity = layer.activity.copy()
    
    layer.update_activity(cycled_point)
    cycled_activity = layer.activity.copy()
    
    # Verify activation patterns are nearly identical
    np.testing.assert_allclose(reference_activity, cycled_activity, rtol=1e-5)


@pytest.mark.parametrize("scale", [0.2])
def test_grid_cell_toroid_y_cycling(layer):
    """Test that grid cell activation cycles correctly when moving along y-axis."""
    # Get height and spacing
    height = layer.height
    spacing = layer.spacing
    
    # Define reference point and a point one full period away on y-axis
    reference_point = (0.5, 0.5)
    cycled_point = (0.5, 0.5 + height * spacing)
    
    # Measure activations at both points
    layer.update_activity(reference_point)
    reference_activity = layer.activity.copy()
    
    layer.update_activity(cycled_point)
    cycled_activity = layer.activity.copy()
    
    # Verify activation patterns are nearly identical
    np.testing.assert_allclose(reference_activity, cycled_activity, rtol=1e-5)


@pytest.mark.parametrize("scale", [0.2])
def test_grid_cell_toroid_diagonal_cycling(layer):
    """Test that grid cell activation cycles correctly when moving diagonally."""
    # Get width, height and spacing
    width, height = layer.width, layer.height
    spacing = layer.spacing
    
    # Define reference point and a point one full period away diagonally
    reference_point = (0.5, 0.5)
    cycled_point = (0.5 + width * spacing, 0.5 + height * spacing)
    
    # Measure activations at both points
    layer.update_activity(reference_point)
    reference_activity = layer.activity.copy()
    
    layer.update_activity(cycled_point)
    cycled_activity = layer.activity.copy()
    
    # Verify activation patterns are nearly identical
    np.testing.assert_allclose(reference_activity, cycled_activity, rtol=1e-5)


@pytest.mark.parametrize("distance_factor", [1, 2, 3])
def test_grid_cell_multiple_cycles(layer, distance_factor):
    """Test that grid cell activation repeats after multiple cycles."""
    # Define reference point
    reference_point = (0.5, 0.5)
    
    # Move by multiple periods
    width = layer.width
    spacing = layer.spacing
    cycled_point = (0.5 + distance_factor * width * spacing, 0.5)
    
    # Measure activations
    layer.update_activity(reference_point)
    reference_activity = layer.activity.copy()
    
    layer.update_activity(cycled_point)
    cycled_activity = layer.activity.copy()
    
    # Patterns should be nearly identical regardless of how many cycles
    np.testing.assert_allclose(reference_activity, cycled_activity, rtol=1e-5)


def test_continuous_movement(layer, position_gen):
    """Test that activation changes smoothly and cycles correctly with continuous movement."""
    # Generate a straight path crossing multiple grid periods
    start = (0.0, 0.5)
    end = (2.5 * layer.width * layer.spacing, 0.5)
    positions = position_gen(start, end, 50)
    
    # Collect activities along the path
    activities = []
    for pos in positions:
        layer.update_activity(pos)
        activities.append(layer.activity.copy())
    
    # Check for smooth transitions between consecutive positions
    correlations = []
    for i in range(len(activities) - 1):
        corr = np.corrcoef(activities[i].flatten(), activities[i+1].flatten())[0, 1]
        correlations.append(corr)
    
    # Average correlation should be high for smooth transitions
    assert np.mean(correlations) > 0.85
    
    # Check periodicity - find points separated by approximately one grid period
    grid_period = layer.width * layer.spacing
    expected_period_index = int(len(positions) * (grid_period / (end[0] - start[0])))
    
    # Correlation should be high between patterns separated by one period
    period_correlation = np.corrcoef(
        activities[0].flatten(), 
        activities[expected_period_index].flatten()
    )[0, 1]
    
    # Verify pattern repeats after one grid period
    assert period_correlation > 0.7
