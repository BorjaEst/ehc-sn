import numpy as np
import pytest
from ehc_sn.mec import GridCellsLayer, MECNetwork


def test_network_initialization():
    """Test that network initializes correctly with multiple grid cell layers."""
    # Create a network with three layers of different scales
    network = MECNetwork(
        [
            GridCellsLayer(width=10, height=10, spacing=0.2, orientation=0.0),
            GridCellsLayer(width=8, height=8, spacing=0.4, orientation=np.pi/6),
            GridCellsLayer(width=6, height=6, spacing=0.8, orientation=np.pi/4)
        ],
        noise_level=0.05
    )
    
    # Verify network properties
    assert len(network.grid_cells) == 3
    assert network.noise_level == 0.05
    assert network.grid_cells[0].spacing == 0.2
    assert network.grid_cells[1].spacing == 0.4
    assert network.grid_cells[2].spacing == 0.8


def test_position_update_propagation(network):
    """Test that position updates propagate to all layers."""
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
        dx = min(abs(max_position[0] - test_position[0]),
                layer.width * layer.spacing - abs(max_position[0] - test_position[0]))
        dy = min(abs(max_position[1] - test_position[1]),
                layer.height * layer.spacing - abs(max_position[1] - test_position[1]))
                
        distance = np.sqrt(dx**2 + dy**2)
        assert distance < layer.spacing * 1.5


def test_noise_effect():
    """Test the effect of different noise levels on network activity."""
    # Create two networks with different noise levels
    low_noise_network = MECNetwork(
        [GridCellsLayer(width=10, height=10, spacing=0.2, orientation=0.0)],
        noise_level=0.01
    )
    
    high_noise_network = MECNetwork(
        [GridCellsLayer(width=10, height=10, spacing=0.2, orientation=0.0)],
        noise_level=0.2
    )
    
    # Update position
    test_position = (1.0, 1.0)
    low_noise_network.set_position(test_position)
    high_noise_network.set_position(test_position)
    
    # Compute variance in activity as an indirect measure of noise
    low_noise_variance = np.var(low_noise_network.grid_cells[0].activity)
    high_noise_variance = np.var(high_noise_network.grid_cells[0].activity)
    
    # Higher noise level should result in higher variance
    assert high_noise_variance > low_noise_variance


def test_position_transition(network, position_gen):
    """Test network response to position transitions."""
    # Generate a path of positions
    start_pos = (0.0, 0.0)
    end_pos = (2.0, 2.0)
    positions = position_gen(start_pos, end_pos, 10)
    
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
            next_activity = activities[i+1][layer_idx].flatten()
            
            # Calculate correlation between consecutive activities
            corr = np.corrcoef(curr_activity, next_activity)[0, 1]
            
            # Correlation should be high for small position changes
            assert corr > 0.5


def test_multiple_scales_integration(network):
    """Test that network properly integrates multiple grid scales."""
    # Set a position and get activities
    test_position = (1.0, 1.0)
    network.set_position(test_position)
    
    # Get peak activity positions for each layer
    peak_positions = []
    for layer in network.grid_cells:
        max_idx = np.unravel_index(np.argmax(layer.activity), layer.activity.shape)
        peak_positions.append(layer.positions[max_idx])
    
    # Different scales should give complementary spatial information
    # We verify this by checking that peak positions are different
    for i in range(len(peak_positions)):
        for j in range(i+1, len(peak_positions)):
            # Distance between peak positions should reflect their scale differences
            dist = np.linalg.norm(np.array(peak_positions[i]) - np.array(peak_positions[j]))
            assert dist > 0.01  # Small threshold to account for discretization
            
    # Test that a new position creates different activity patterns
    new_position = (1.5, 1.5)
    network.set_position(new_position)
    
    # Check that activity patterns changed for all layers
    for i, layer in enumerate(network.grid_cells):
        max_idx = np.unravel_index(np.argmax(layer.activity), layer.activity.shape)
        new_peak = layer.positions[max_idx]
        
        # Verify the peak shifted
        assert new_peak != peak_positions[i]


def test_orientation_effect():
    """Test the effect of different orientations on grid cell patterns."""
    # Create networks with different orientations
    orientations = [0.0, np.pi/6, np.pi/4]
    networks = []
    
    for orientation in orientations:
        networks.append(MECNetwork(
            [GridCellsLayer(width=10, height=10, spacing=0.3, orientation=orientation)],
            noise_level=0.0  # No noise for clearer comparison
        ))
    
    # Set the same position for all networks
    test_position = (1.0, 1.0)
    for network in networks:
        network.set_position(test_position)
    
    # Compare activity patterns - they should be different due to orientation
    for i in range(len(networks)):
        for j in range(i+1, len(networks)):
            activity_i = networks[i].grid_cells[0].activity.flatten()
            activity_j = networks[j].grid_cells[0].activity.flatten()
            
            # Calculate correlation
            corr = np.corrcoef(activity_i, activity_j)[0, 1]
            
            # Correlation should be lower for different orientations
            assert corr < 0.9

