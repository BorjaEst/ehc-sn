#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified DRTP (Direct Random Target Projection) Script

Merged and simplified version of the DRTP modules with fixed parameters:
- Optimizer: NAG (SGD with Nesterov) with lr=1e-4
- Loss: BCE (Binary Cross Entropy)
- Training mode: DRTP only
- Topology: CONV_32_5_1_2_FC_1000_FC_10
- Output activation: Sigmoid
- Conv activation: Tanh
- Dataset: MNIST only
- No dropout, no regression, single trial

Based on: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
Fixed random learning signals allow for feedforward training of deep neural networks,"
Frontiers in Neuroscience, vol. 15, no. 629892, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm

from ehc_sn.models import encoders

# -------------------------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------------------------


def load_mnist_data():
    """Load MNIST dataset with fixed parameters"""
    # Fixed parameters
    batch_size = 100
    test_batch_size = 1000

    kwargs = {"num_workers": 2, "pin_memory": True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    traintest_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]),
        ),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]),
        ),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, traintest_loader, test_loader


# -------------------------------------------------------------------------------------------
# Training Functions
# -------------------------------------------------------------------------------------------


def train_epoch(model, device, train_loader, optimizer):
    """Train for one epoch"""
    model.train()

    for batch_idx, (data, label) in enumerate(tqdm(train_loader, desc="Training")):
        data, label = data.to(device), label.to(device)

        # Convert labels to one-hot encoding for BCE loss
        targets = torch.zeros(label.shape[0], 10, device=device).scatter_(1, label.unsqueeze(1), 1.0)

        optimizer.zero_grad()
        output = model(data, targets)
        loss_val = F.binary_cross_entropy(output, targets)
        loss_val.backward()
        optimizer.step()


def test_epoch(model, device, test_loader, phase):
    """Test the model"""
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            targets = torch.zeros(label.shape[0], 10, device=device).scatter_(1, label.unsqueeze(1), 1.0)

            output = model(data, None)
            test_loss += F.binary_cross_entropy(output, targets, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    loss = test_loss / len_dataset
    acc = 100.0 * correct / len_dataset
    print(f"\t[{phase:>5}ing set] Loss: {loss:6f}, Accuracy: {acc:6.2f}%")


# -------------------------------------------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------------------------------------------


def main():
    """Main function to run simplified DRTP training"""
    # Fixed parameters
    epochs = 100
    lr = 1e-4

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, traintest_loader, test_loader = load_mnist_data()

    # Initialize model
    print("Initializing DRTP network...")
    torch.manual_seed(42)
    model = encoders.DRTPConv2D(
        encoders.EncoderParams(
            input_shape=(1, 28, 28),  # MNIST input shape
            latent_dim=10,  # Number of classes for MNIST
            activation_fn=nn.Tanh,
        )
    )

    if torch.cuda.is_available():
        model.cuda()

    print("=== Model ===")
    print(model)

    # Initialize optimizer (NAG - SGD with Nesterov)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    print(f"\n=== Starting model training with {epochs} epochs:\n")

    # Training loop
    for epoch in range(1, epochs + 1):
        # Training
        train_epoch(model, device, train_loader, optimizer)

        # Testing
        print(f"\nSummary of epoch {epoch}:")
        test_epoch(model, device, traintest_loader, "Train")
        test_epoch(model, device, test_loader, "Test")


if __name__ == "__main__":
    main()
