import numpy as np
import pytest
from ehc_sn.mec import GridCellsLayer, MECNetwork


@pytest.mark.parametrize("noise_level", [0.0, 0.2])
def test_network_initialization(network_gen, noise_level):
    """Test that network initializes correctly with multiple grid cell layers."""
    network = network_gen(noise_level=noise_level)
    assert len(network.grid_cells) == 3
    assert network.noise_level == noise_level
    assert network.grid_cells[0].spacing == 0.2
    assert network.grid_cells[0].orientation == 0.0
    assert network.grid_cells[1].spacing == 0.4
    assert network.grid_cells[1].orientation == 0.0
    assert network.grid_cells[2].spacing == 0.8
    assert network.grid_cells[2].orientation == 0.0


def test_position_update_propagation(network_gen):
    """Test that position updates propagate to all layers."""
    network = network_gen()

    # Initial position
    test_position = (1.0, 1.0)
    network.set_position(test_position)

    # Verify all layers have been updated
    for layer in network.grid_cells:
        # Check if activity has been updated by ensuring some cells are active
        assert np.max(layer.activity) > 0.0

        # Find the most active cell and verify it corresponds to the position
        max_idx = np.unravel_index(np.argmax(layer.activity), layer.activity.shape)
        max_position = layer.positions[max_idx]

        # Due to toroidal wrapping and periodic nature, we check that the most active
        # cell is within a reasonable distance (considering scale)
        dx = min(
            abs(max_position[0] - test_position[0]),
            layer.width * layer.spacing - abs(max_position[0] - test_position[0]),
        )
        dy = min(
            abs(max_position[1] - test_position[1]),
            layer.height * layer.spacing - abs(max_position[1] - test_position[1]),
        )

        distance = np.sqrt(dx**2 + dy**2)
        assert distance < layer.spacing * 1.5


@pytest.mark.parametrize("noise_levels", [{"high": 0.2, "low": 0.01}])
def test_noise_effect(network_gen, noise_levels):
    """Test the effect of different noise levels on network activity."""
    # Create two networks with different noise levels
    low_noise_network = network_gen(noise_level=noise_levels["low"])
    high_noise_network = network_gen(noise_level=noise_levels["high"])

    # Update position
    test_position = (1.0, 1.0)
    low_noise_network.set_position(test_position)
    high_noise_network.set_position(test_position)

    # Compute variance in activity as an indirect measure of noise
    low_noise_variance = np.var(low_noise_network.grid_cells[0].activity)
    high_noise_variance = np.var(high_noise_network.grid_cells[0].activity)

    # Higher noise level should result in higher variance
    assert high_noise_variance > low_noise_variance


@pytest.mark.parametrize("num_points", [40, 60])
def test_position_transition(network_gen, position_gen, num_points):
    """Test network response to position transitions."""
    network = network_gen()

    # Generate a path of positions
    start_pos = (0.0, 0.0)
    end_pos = (1.0, 1.0)
    positions = position_gen(start_pos, end_pos, num_points)

    # Track activities along the path
    activities = []
    for pos in positions:
        network.set_position(pos)
        # Store a copy of each layer's activity
        layer_activities = [layer.activity.copy() for layer in network.grid_cells]
        activities.append(layer_activities)

    # Check for smooth transitions between positions
    for layer_idx in range(len(network.grid_cells)):
        for i in range(len(activities) - 1):
            curr_activity = activities[i][layer_idx].flatten()
            next_activity = activities[i + 1][layer_idx].flatten()

            # Calculate correlation between consecutive activities
            corr = np.corrcoef(curr_activity, next_activity)[0, 1]

            # Correlation should be high for small position changes
            assert corr > 0.5


def test_multiple_scales_integration(network_gen, layer_gen):
    """Test that network properly integrates multiple grid scales."""
    network = network_gen([layer_gen(spacing=x) for x in [0.2, 0.4, 0.8]])

    # Set a position and get activities
    test_position = (1.0, 1.0)
    network.set_position(test_position)

    # Get peak activity positions for each layer
    peak_positions = []
    for layer in network.grid_cells:
        max_idx = np.unravel_index(np.argmax(layer.activity), layer.activity.shape)
        peak_positions.append(layer.positions[max_idx])

    # Different scales should give complementary spatial information
    # We verify this by checking that the grid patterns are unique across scales
    # Note: Peak positions can legitimately overlap sometimes, so instead we'll
    # check that activity patterns across scales are different
    for i in range(len(network.grid_cells)):
        for j in range(i + 1, len(network.grid_cells)):
            activity_i = network.grid_cells[i].activity.flatten()
            activity_j = network.grid_cells[j].activity.flatten()

            # Calculate correlation between activities of different scales
            corr = np.corrcoef(activity_i, activity_j)[0, 1]

            # Different scales should not have identical activity patterns
            # A perfect correlation of 1.0 would indicate identical patterns
            assert corr < 0.99

    # Test that a new position creates different activity patterns
    new_position = (1.5, 1.5)
    network.set_position(new_position)

    # Check that activity patterns changed for all layers
    for i, layer in enumerate(network.grid_cells):
        max_idx = np.unravel_index(np.argmax(layer.activity), layer.activity.shape)
        new_peak = layer.positions[max_idx]

        # Get the previous activity pattern for comparison
        # Set position back to original position to get old activity
        network.set_position(test_position)
        old_activity = layer.activity.copy()

        # Set position back to new position
        network.set_position(new_position)
        new_activity = layer.activity.copy()

        # Compare activity patterns to ensure they changed
        correlation = np.corrcoef(old_activity.flatten(), new_activity.flatten())[0, 1]

        # Different positions should result in different activity patterns
        # A correlation significantly below 1.0 indicates pattern change
        assert correlation < 0.95


def test_orientation_effect(network_gen, layer_gen):
    """Test the effect of different orientations on grid cell patterns."""
    # Create networks with different orientations
    orientations = [0.0, np.pi / 6, np.pi / 4]
    networks = [network_gen([layer_gen(orientation=x)]) for x in orientations]

    # Set the same position for all networks
    test_position = (1.0, 1.0)
    for network in networks:
        network.set_position(test_position)

    # Compare activity patterns - they should be different due to orientation
    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            activity_i = networks[i].grid_cells[0].activity.flatten()
            activity_j = networks[j].grid_cells[0].activity.flatten()

            # Calculate correlation
            corr = np.corrcoef(activity_i, activity_j)[0, 1]

            # Correlation should be lower for different orientations
            assert corr < 0.9
