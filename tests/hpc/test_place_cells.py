import itertools

import numpy as np
import pytest
from scipy.spatial.distance import cdist


class TestInitialization:
    """Test the initialization of PlaceCellsLayer."""

    def test_default_initialization(self, layer_gen):
        """Test default initialization of PlaceCellsLayer."""
        layer = layer_gen()

        # Check default parameters
        assert layer.n_cells == 100
        assert layer.field_size == 0.3
        assert layer.environment_size == (10.0, 10.0)
        assert layer.centers.shape == (100, 2)
        assert np.all(layer.centers[:, 0] >= 0) and np.all(layer.centers[:, 0] <= 10.0)
        assert np.all(layer.centers[:, 1] >= 0) and np.all(layer.centers[:, 1] <= 10.0)
        assert layer.activity.shape == (100,)

    @pytest.mark.parametrize("n_cells", [50, 200, 500])
    def test_custom_n_cells(self, layer_gen, n_cells):
        """Test initialization with custom number of cells."""
        layer = layer_gen(n_cells=n_cells)

        # Check that the number of cells matches
        assert layer.n_cells == n_cells
        assert layer.centers.shape == (n_cells, 2)
        assert layer.activity.shape == (n_cells,)

    @pytest.mark.parametrize("field_size", [0.1, 0.3, 0.5])
    def test_custom_field_size(self, layer_gen, field_size):
        """Test initialization with custom field size."""
        layer = layer_gen(field_size=field_size)

        # Check that the field size matches
        assert layer.field_size == field_size
        assert np.all(layer.centers[:, 0] >= 0)
        assert np.all(layer.centers[:, 0] <= layer.environment_size[0])
        assert np.all(layer.centers[:, 1] >= 0)
        assert np.all(layer.centers[:, 1] <= layer.environment_size[1])

    @pytest.mark.parametrize("env_size", [(10.0, 10.0), (20.0, 20.0)])
    def test_custom_environment_size(self, layer_gen, env_size):
        """Test initialization with custom environment size."""
        layer = layer_gen(environment_size=env_size)

        # Check that the environment size matches
        assert layer.environment_size == env_size
        assert np.all(layer.centers[:, 0] >= 0)
        assert np.all(layer.centers[:, 0] <= env_size[0])
        assert np.all(layer.centers[:, 1] >= 0)
        assert np.all(layer.centers[:, 1] <= env_size[1])

    def test_initial_activity(self, layer_gen):
        """Test that initial activity is zero."""
        layer = layer_gen()
        assert np.all(layer.activity == 0)


@pytest.mark.parametrize("position", [(5.0, 5.0), (2.0, 8.0)])
class TestActivityCalculation:
    """Test the activity calculation of PlaceCellsLayer."""

    def test_activity_range(self, layer_gen, position):
        """Test that activity values are in the range [0, 1]."""
        layer = layer_gen()
        activity = layer.calculate_activity(position)
        assert np.all(activity >= 0) and np.all(activity <= 1)

    def test_activity_shape(self, layer_gen, position):
        """Test that activity has the correct shape."""
        layer = layer_gen()
        activity = layer.calculate_activity(position)
        assert activity.shape == (layer.n_cells,)

    def test_activity_decrease(self, layer_gen, position):
        """Test that activity decreases with distance from cell centers."""
        layer = layer_gen()
        activity = layer.calculate_activity(position)

        # Calculate distances from the position to each cell center
        distances = cdist([position], layer.centers)[0]
        # Sort cells by distance
        sorted_indices = np.argsort(distances)

        # Check if activity generally decreases with distance
        closest_cells = sorted_indices[:10]
        farthest_cells = sorted_indices[-10:]
        assert np.mean(activity[closest_cells]) > np.mean(activity[farthest_cells])


class TestSetPosition:
    """Test the set_position method of PlaceCellsLayer."""

    @pytest.mark.parametrize("position", [(5.0, 5.0), (2.0, 8.0)])
    def test_set_position(self, layer_gen, position):
        """Test setting position updates activity correctly."""
        layer = layer_gen()

        # After setting position, activity should be updated
        result = layer.set_position(position)
        assert np.array_equal(result, layer.activity)
        assert not np.all(layer.activity == 0)

        # Activity should match calculated activity
        expected = layer.calculate_activity(position)
        assert np.array_equal(layer.activity, expected)

    def test_patterns(self, layer_gen, position_gen):
        """Test that different positions produce different activity patterns."""
        layer = layer_gen()
        positions = position_gen(start=(0, 0), end=(10, 10), num_points=10)
        activities = [layer.set_position(pos) for pos in positions]

        # Activities should be different for different positions
        for activity1, activity2 in itertools.combinations(activities, 2):
            assert not np.array_equal(activity1, activity2)
            assert np.sum(np.abs(activity1 - activity2)) > 0


class TestFieldEffects:
    """Test the effects of field size and cell density on place cell activity."""

    @pytest.mark.parametrize("position", [(5.0, 5.0), (2.0, 8.0)])
    def test_field_size(self, layer_gen, position):
        """Test that larger fields result in more active cells."""
        small_field = layer_gen(field_size=0.2)
        large_field = layer_gen(field_size=0.5)

        # Use the same centers for both layers to isolate field size effect
        large_field.centers = small_field.centers.copy()
        small_activity = small_field.calculate_activity(position)
        large_activity = large_field.calculate_activity(position)

        # Larger fields should result in more cells being active
        assert np.sum(large_activity > 0.1) > np.sum(small_activity > 0.1)

        # Calculate average activity for cells at different distances
        distances = cdist([position], small_field.centers)[0]
        far_cells = distances > 1.0

        # Larger fields should have higher activity for distant cells
        assert np.mean(large_activity[far_cells]) > np.mean(small_activity[far_cells])
