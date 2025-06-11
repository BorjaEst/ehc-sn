import numpy as np
import pytest


class TestInitialization:
    """Test the initialization of HPCNetwork."""

    def test_default_initialization(self, layer_gen, network_gen):
        """Test default initialization of HPCNetwork."""
        place_cells = layer_gen()
        network = network_gen(place_cells)

        # Check attributes
        assert network.place_cells is place_cells
        assert network.noise_level == 0.1
        assert network.memory_decay == 0.05
        assert network.current_position is None
        assert network.memory_trace == []

    @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.3])
    def test_custom_noise_level(self, network_gen, noise_level):
        """Test initialization with custom noise level."""
        network = network_gen(noise_level=noise_level)

        # Check that noise level matches
        assert network.noise_level == noise_level

    @pytest.mark.parametrize("memory_decay", [0.01, 0.05, 0.1])
    def test_custom_memory_decay(self, network_gen, memory_decay):
        """Test initialization with custom memory decay."""
        network = network_gen(memory_decay=memory_decay)

        # Check that memory decay matches
        assert network.memory_decay == memory_decay


@pytest.mark.parametrize("position", [(5.0, 5.0), (2.0, 8.0)])
class TestPositionSetting:
    """Test position setting and related activity in HPCNetwork."""

    def test_set_position(self, network_gen, position):
        """Test setting position updates current_position attribute."""
        network = network_gen()

        network.set_position(position)
        assert np.array_equal(network.current_position, np.array(position))

    def test_activity_with_noise(self, network_gen, position):
        """Test that activity includes noise component."""
        network = network_gen()

        # Get activity with noise
        activity_with_noise = network.set_position(position)

        # Get pure activity without noise
        pure_activity = network.place_cells.calculate_activity(position)

        # Activity should differ due to noise
        assert not np.array_equal(activity_with_noise, pure_activity)

        # But should still be within reasonable bounds
        assert np.all((activity_with_noise >= 0) & (activity_with_noise <= 1))


class TestMemoryFunctions:
    """Test memory storage and recall features of HPCNetwork."""

    def test_store_memory(self, network_gen):
        """Test storing memory updates memory trace."""
        network = network_gen()

        # Store memory at two positions
        network.set_position((3.0, 3.0))
        network.store_memory()

        network.set_position((7.0, 7.0))
        network.store_memory()

        # Memory trace should contain two entries
        assert len(network.memory_trace) == 2

        # Each entry should have position and activity
        for entry in network.memory_trace:
            assert "position" in entry
            assert "activity" in entry
            assert isinstance(entry["position"], np.ndarray)
            assert isinstance(entry["activity"], np.ndarray)

    @pytest.mark.parametrize(
        "positions", [[(3.0, 3.0), (7.0, 7.0)], [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]]
    )
    def test_recall_memory(self, network_gen, positions):
        """Test recalling memory returns correct stored pattern."""
        network = network_gen()

        # Store activities for multiple positions
        stored_activities = []
        for pos in positions:
            network.set_position(pos)
            stored_activities.append(network.place_cells.activity.copy())
            network.store_memory()

        # Check that each stored activity recalls the correct memory
        for i, activity in enumerate(stored_activities):
            recalled = network.recall_memory(activity)
            assert np.array_equal(recalled["position"], np.array(positions[i]))
            assert np.array_equal(recalled["activity"], stored_activities[i])


class TestPatternOverlap:
    """Test pattern overlap calculations in HPCNetwork."""

    def test_identical_patterns(self, network_gen):
        """Test overlap between identical patterns."""
        network = network_gen()
        pattern = np.array([0.5, 0.7, 0.2, 0.9, 0.3])

        # Overlap with self should be 1.0
        assert network.calculate_overlap(pattern, pattern) > 0.99

    def test_orthogonal_patterns(self, network_gen):
        """Test overlap between orthogonal patterns."""
        network = network_gen()
        pattern1 = np.array([1.0, 0.0, 1.0, 0.0])
        pattern2 = np.array([0.0, 1.0, 0.0, 1.0])

        # Orthogonal patterns should have zero overlap
        assert network.calculate_overlap(pattern1, pattern2) == 0.0

    def test_partial_overlap(self, network_gen):
        """Test overlap between partially similar patterns."""
        network = network_gen()
        pattern1 = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        pattern2 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        # Calculate and check overlap
        overlap = network.calculate_overlap(pattern1, pattern2)
        assert 0.0 < overlap < 1.0


class TestGridCellIntegration:
    """Test integration with grid cell input."""

    @pytest.mark.parametrize("position", [(4.0, 6.0), (2.0, 3.0)])
    def test_with_position(self, network_gen, position):
        """Test updating from grid cells with position set."""
        network = network_gen()
        network.set_position(position)

        # Create mock grid cell activity
        mock_grid_activity = np.random.rand(100)

        # Update from grid cells
        result = network.update_from_grid_cells(mock_grid_activity)

        # Should return and update place cell activity
        assert np.array_equal(result, network.place_cells.activity)
        assert np.array_equal(network.current_position, np.array(position))

    def test_without_position(self, network_gen):
        """Test updating from grid cells without position set."""
        network = network_gen()

        # Create mock grid cell activity
        mock_grid_activity = np.random.rand(100)

        # Update from grid cells without position set
        result = network.update_from_grid_cells(mock_grid_activity)

        # Should return current activity, position still None
        assert np.array_equal(result, network.place_cells.activity)
        assert network.current_position is None
