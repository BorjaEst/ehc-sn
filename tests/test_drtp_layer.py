import pytest
import torch
from torch import nn

from ehc_sn.core.drtp import DRTPFunction, DRTPLayer, drtp_layer


class TestDRTPLayer:
    """Test suite for Direct Random Target Projection (DRTP) layer implementation."""

    # -------------------------------------------------------------------------------------------
    def test_drtp_forward_pass(self):
        """Test that DRTP forward pass returns input unchanged."""
        input_tensor = torch.randn(2, 4, requires_grad=True)
        B = torch.randn(3, 4) * 0.1
        target = torch.randn(2, 3)

        output = drtp_layer(input_tensor, B, target)

        # Forward pass should return input unchanged
        assert torch.allclose(output, input_tensor)
        assert output.requires_grad == input_tensor.requires_grad

    # -----------------------------------------------------------------------------------
    def test_drtp_gradient_shape(self):
        """Test that DRTP gradients have correct shapes."""
        batch_size, hidden_dim, target_dim = 2, 4, 3

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        B = torch.randn(target_dim, hidden_dim) * 0.1
        target = torch.randn(batch_size, target_dim)

        output = drtp_layer(input_tensor, B, target)
        loss = output.sum()
        loss.backward()

        # Check gradient shapes
        assert input_tensor.grad.shape == input_tensor.shape
        assert input_tensor.grad.shape == (batch_size, hidden_dim)

    # -----------------------------------------------------------------------------------
    def test_drtp_gradient_computation(self):
        """Test that DRTP computes gradients correctly using the formula: delta = target @ B."""
        batch_size, hidden_dim, target_dim = 1, 3, 2

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        B = torch.randn(target_dim, hidden_dim) * 0.1
        target = torch.ones(batch_size, target_dim)  # Use ones for predictable results

        output = drtp_layer(input_tensor, B, target)
        loss = output.sum()
        loss.backward()

        # Manual computation of DRTP gradient
        expected_grad = torch.matmul(target, B)  # (batch_size, target_dim) @ (target_dim, hidden_dim)

        assert torch.allclose(input_tensor.grad, expected_grad, atol=1e-6)

    # -----------------------------------------------------------------------------------
    def test_drtp_with_linear_layer(self):
        """Test DRTP with a linear layer to ensure gradients propagate to weights."""
        input_size, hidden_size, target_size = 4, 3, 2
        batch_size = 1

        # Create layer and inputs
        layer = torch.nn.Linear(input_size, hidden_size)
        x = torch.randn(batch_size, input_size, requires_grad=True)
        B = torch.randn(target_size, hidden_size) * 0.1
        target = torch.ones(batch_size, target_size)

        # Forward pass
        hidden = layer(x)
        drtp_output = drtp_layer(hidden, B, target)
        loss = drtp_output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

        # Check gradient shapes
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    # -----------------------------------------------------------------------------------
    def test_drtp_multiple_layers(self):
        """Test DRTP with multiple layers in sequence."""
        # Layer dimensions
        input_size, hidden1_size, hidden2_size = 4, 3, 2
        target1_size, target2_size = 2, 1
        batch_size = 1

        # Create layers
        layer1 = torch.nn.Linear(input_size, hidden1_size)
        layer2 = torch.nn.Linear(hidden1_size, hidden2_size)

        # Random projection matrices
        B1 = torch.randn(target1_size, hidden1_size) * 0.1
        B2 = torch.randn(target2_size, hidden2_size) * 0.1

        # Inputs and targets
        x = torch.randn(batch_size, input_size, requires_grad=True)
        target1 = torch.ones(batch_size, target1_size)
        target2 = torch.ones(batch_size, target2_size)

        # Forward pass
        y1 = layer1(x)
        y1_drtp = drtp_layer(y1, B1, target1)

        y2 = layer2(y1_drtp)
        y2_drtp = drtp_layer(y2, B2, target2)

        loss = y2_drtp.sum()
        loss.backward()

        # Check that all layers have gradients
        assert layer1.weight.grad is not None
        assert layer1.bias.grad is not None
        assert layer2.weight.grad is not None
        assert layer2.bias.grad is not None
        assert x.grad is not None

    # -----------------------------------------------------------------------------------
    def test_drtp_gradient_independence(self):
        """Test that DRTP gradients are independent of standard gradients."""
        batch_size, hidden_dim, target_dim = 1, 3, 2

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        B = torch.randn(target_dim, hidden_dim) * 0.1
        target = torch.ones(batch_size, target_dim)

        # DRTP backward pass
        output = drtp_layer(input_tensor, B, target)
        loss = output.sum()
        loss.backward()

        drtp_grad = input_tensor.grad.clone()

        # Reset gradients
        input_tensor.grad.zero_()

        # Standard backward pass (without DRTP)
        input_tensor2 = input_tensor.clone().detach().requires_grad_(True)
        standard_loss = input_tensor2.sum()
        standard_loss.backward()

        standard_grad = input_tensor2.grad

        # DRTP gradient should be different from standard gradient
        assert not torch.allclose(drtp_grad, standard_grad)

        # DRTP gradient should match manual computation
        expected_drtp_grad = torch.matmul(target, B)
        assert torch.allclose(drtp_grad, expected_drtp_grad, atol=1e-6)

    # -----------------------------------------------------------------------------------
    def test_drtp_batch_consistency(self):
        """Test that DRTP works consistently across different batch sizes."""
        hidden_dim, target_dim = 3, 2
        B = torch.randn(target_dim, hidden_dim) * 0.1

        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
            target = torch.ones(batch_size, target_dim)

            output = drtp_layer(input_tensor, B, target)
            loss = output.sum()
            loss.backward()

            # Check gradient shapes
            assert input_tensor.grad.shape == (batch_size, hidden_dim)

            # Check gradient computation
            expected_grad = torch.matmul(target, B)
            assert torch.allclose(input_tensor.grad, expected_grad, atol=1e-6)

    # -----------------------------------------------------------------------------------
    def test_drtp_zero_target(self):
        """Test DRTP behavior with zero target."""
        batch_size, hidden_dim, target_dim = 1, 3, 2

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        B = torch.randn(target_dim, hidden_dim) * 0.1
        target = torch.zeros(batch_size, target_dim)  # Zero target

        output = drtp_layer(input_tensor, B, target)
        loss = output.sum()
        loss.backward()

        # With zero target, gradient should be zero
        expected_grad = torch.zeros_like(input_tensor)
        assert torch.allclose(input_tensor.grad, expected_grad, atol=1e-6)

    # -----------------------------------------------------------------------------------
    def test_drtp_random_matrix_effect(self):
        """Test that different random matrices B produce different gradients."""
        batch_size, hidden_dim, target_dim = 1, 3, 2

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        target = torch.ones(batch_size, target_dim)

        # Two different random matrices
        B1 = torch.randn(target_dim, hidden_dim) * 0.1
        B2 = torch.randn(target_dim, hidden_dim) * 0.1

        # Ensure they are different
        assert not torch.allclose(B1, B2)

        # Test with first matrix
        output1 = drtp_layer(input_tensor, B1, target)
        loss1 = output1.sum()
        loss1.backward()
        grad1 = input_tensor.grad.clone()

        # Reset gradients and test with second matrix
        input_tensor.grad.zero_()
        output2 = drtp_layer(input_tensor, B2, target)
        loss2 = output2.sum()
        loss2.backward()
        grad2 = input_tensor.grad.clone()

        # Different matrices should produce different gradients
        assert not torch.allclose(grad1, grad2)


# -------------------------------------------------------------------------------------------
class TestDRTPFunction:
    """Test the underlying DRTP autograd function directly."""

    # -----------------------------------------------------------------------------------
    def test_drtp_function_forward(self):
        """Test DRTPFunction forward method."""
        input_tensor = torch.randn(2, 4)
        B = torch.randn(3, 4)
        target = torch.randn(2, 3)

        output = DRTPFunction.apply(input_tensor, B, target)

        assert torch.allclose(output, input_tensor)

    # -----------------------------------------------------------------------------------
    def test_drtp_function_gradient_implementation(self):
        """Test that DRTPFunction correctly implements the DRTP gradient formula."""
        input_tensor = torch.randn(1, 3, requires_grad=True, dtype=torch.double)
        B = torch.randn(2, 3, dtype=torch.double)
        target = torch.randn(1, 2, dtype=torch.double)

        # Forward and backward pass
        output = DRTPFunction.apply(input_tensor, B, target)
        loss = output.sum()
        loss.backward()

        # Check that gradient matches DRTP formula: target @ B
        expected_grad = torch.matmul(target, B)
        assert torch.allclose(input_tensor.grad, expected_grad, atol=1e-6)

        # Note: DRTP intentionally does NOT match numerical gradients because
        # it uses a fixed random projection instead of the true gradient.
        # This is the core feature of DRTP - it provides learning signals
        # that are biologically plausible but mathematically different from backprop.


# -------------------------------------------------------------------------------------------
class TestDRTPLayerModule:
    """Test suite for the DRTPLayer nn.Module class."""

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_initialization(self):
        """Test DRTPLayer module initialization."""
        target_dim, hidden_dim = 5, 10
        layer = DRTPLayer(target_dim=target_dim, hidden_dim=hidden_dim)

        assert layer.target_dim == target_dim
        assert layer.hidden_dim == hidden_dim
        assert layer.B.shape == (target_dim, hidden_dim)
        assert isinstance(layer.B, torch.Tensor)

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_forward_pass(self):
        """Test DRTPLayer module forward pass."""
        batch_size, target_dim, hidden_dim = 3, 5, 10
        layer = DRTPLayer(target_dim=target_dim, hidden_dim=hidden_dim)

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        target = torch.randn(batch_size, target_dim)

        output = layer(input_tensor, target)

        # Output should be identical to input
        assert torch.allclose(output, input_tensor)
        assert output.requires_grad == input_tensor.requires_grad

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_gradient_computation(self):
        """Test that DRTPLayer computes gradients correctly."""
        batch_size, target_dim, hidden_dim = 2, 3, 4
        layer = DRTPLayer(target_dim=target_dim, hidden_dim=hidden_dim)

        input_tensor = torch.randn(batch_size, hidden_dim, requires_grad=True)
        target = torch.ones(batch_size, target_dim)  # Use ones for predictable results

        output = layer(input_tensor, target)
        loss = output.sum()
        loss.backward()

        # Expected gradient should be target @ layer.B
        expected_grad = torch.matmul(target, layer.B)

        assert torch.allclose(input_tensor.grad, expected_grad, atol=1e-6)

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_dimension_validation(self):
        """Test that DRTPLayer validates input dimensions."""
        layer = DRTPLayer(target_dim=5, hidden_dim=10)

        # Test with wrong input dimension
        wrong_input = torch.randn(2, 8)  # Should be (2, 10)
        target = torch.randn(2, 5)

        with pytest.raises(ValueError, match="Input last dimension"):
            layer(wrong_input, target)

        # Test with wrong target dimension
        input_tensor = torch.randn(2, 10)
        wrong_target = torch.randn(2, 3)  # Should be (2, 5)

        with pytest.raises(ValueError, match="Target last dimension"):
            layer(input_tensor, wrong_target)

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_reinit_projection_matrix(self):
        """Test reinitializing the projection matrix."""
        layer = DRTPLayer(target_dim=3, hidden_dim=5)
        original_B = layer.B.clone()

        # Reinitialize with different scale
        layer.reinit_projection_matrix(scale=0.5)

        # Matrix should be different
        assert not torch.allclose(layer.B, original_B)
        assert layer.scale == 0.5

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_extra_repr(self):
        """Test the extra_repr method for debugging."""
        layer = DRTPLayer(target_dim=3, hidden_dim=5, scale=0.2)
        repr_str = layer.extra_repr()

        assert "target_dim=3" in repr_str
        assert "hidden_dim=5" in repr_str
        assert "scale=0.2" in repr_str

    # -----------------------------------------------------------------------------------
    def test_drtp_layer_with_linear_layers(self):
        """Test DRTPLayer integration with linear layers."""
        input_dim, hidden_dim, output_dim = 4, 6, 3
        target_dim = 2

        # Create a small network with DRTP
        linear1 = nn.Linear(input_dim, hidden_dim)
        drtp = DRTPLayer(target_dim=target_dim, hidden_dim=hidden_dim)
        linear2 = nn.Linear(hidden_dim, output_dim)

        x = torch.randn(2, input_dim, requires_grad=True)
        target = torch.randn(2, target_dim)

        # Forward pass
        h = linear1(x)
        h_drtp = drtp(h, target)
        output = linear2(h_drtp)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that all layers have gradients
        assert linear1.weight.grad is not None
        assert linear2.weight.grad is not None
        assert x.grad is not None
