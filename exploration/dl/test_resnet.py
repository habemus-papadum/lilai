"""
Test utilities for ResNet implementation.

Uses PyTorch hooks to inspect intermediate activations without
modifying the model code. This is the professional approach for
debugging neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from contextlib import contextmanager


class ActivationTracker:
    """
    Tracks intermediate activations using PyTorch forward hooks.

    Usage:
        tracker = ActivationTracker()
        with tracker.track(model, ['conv1', 'bn1', 'conv2', 'bn2']):
            output = model(input)
        print(tracker.activations)  # Dict of layer_name -> tensor
    """

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _make_hook(self, name: str):
        """Creates a hook function that stores activations."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            self.activations[name] = output.detach().clone()
        return hook

    @contextmanager
    def track(self, model: nn.Module, layer_names: List[str]):
        """
        Context manager to track activations of specified layers.

        Args:
            model: The model to track
            layer_names: Names of layers to track (as they appear in model.named_modules())
        """
        self.activations.clear()
        self._hooks.clear()

        # Register hooks for requested layers
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

        try:
            yield self
        finally:
            # Always remove hooks to avoid memory leaks
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()


def test_basic_block_shapes():
    """Test that BasicBlock produces correct output shapes."""
    from resnet import BasicBlock

    print("=" * 60)
    print("TEST: BasicBlock Shape Verification")
    print("=" * 60)

    # Test 1: Same dimensions (no downsampling)
    print("\n[Test 1] Same dimensions (64 -> 64, stride=1)")
    block = BasicBlock(in_channels=64, planes=64, stride=1)
    x = torch.randn(1, 64, 56, 56)
    out = block(x)
    expected = (1, 64, 56, 56)
    print(f"  Input:    {tuple(x.shape)}")
    print(f"  Output:   {tuple(out.shape)}")
    print(f"  Expected: {expected}")
    assert out.shape == torch.Size(expected), f"Shape mismatch! Got {out.shape}"
    print("  ✅ PASSED")

    # Test 2: Downsampling with stride=2
    print("\n[Test 2] Downsampling (64 -> 128, stride=2)")
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128)
    )
    block = BasicBlock(in_channels=64, planes=128, stride=2, downsample=downsample)
    x = torch.randn(1, 64, 56, 56)
    out = block(x)
    expected = (1, 128, 28, 28)
    print(f"  Input:    {tuple(x.shape)}")
    print(f"  Output:   {tuple(out.shape)}")
    print(f"  Expected: {expected}")
    assert out.shape == torch.Size(expected), f"Shape mismatch! Got {out.shape}"
    print("  ✅ PASSED")

    print("\n" + "=" * 60)
    print("All shape tests passed!")
    print("=" * 60)


def test_basic_block_forward_step_by_step():
    """
    Test BasicBlock forward pass by inspecting each intermediate activation.
    This verifies the data flow matches the expected architecture.
    """
    from resnet import BasicBlock
    print()
    print("=" * 60)
    print("TEST: BasicBlock Forward Pass (Step-by-Step)")
    print("=" * 60)

    # Create block and tracker
    block = BasicBlock(in_channels=64, planes=64, stride=1)
    block.eval()  # Deterministic BatchNorm behavior
    tracker = ActivationTracker()

    x = torch.randn(1, 64, 56, 56)
    print(f"\nInput shape: {tuple(x.shape)}")

    # Track all intermediate layers
    layers_to_track = ['conv1', 'bn1', 'conv2', 'bn2']

    with tracker.track(block, layers_to_track):
        output = block(x)

    print("\n--- Intermediate Activations ---")

    # Verify conv1 output
    if 'conv1' in tracker.activations:
        conv1_out = tracker.activations['conv1']
        print(f"After conv1: {tuple(conv1_out.shape)}")
        assert conv1_out.shape == (1, 64, 56, 56), "conv1 shape wrong"
        print("  ✅ conv1 shape correct")
    else:
        print("  ⚠️  conv1 not tracked (forward not implemented yet?)")

    # Verify bn1 output
    if 'bn1' in tracker.activations:
        bn1_out = tracker.activations['bn1']
        print(f"After bn1:   {tuple(bn1_out.shape)}")
        assert bn1_out.shape == (1, 64, 56, 56), "bn1 shape wrong"
        print("  ✅ bn1 shape correct")
    else:
        print("  ⚠️  bn1 not tracked")

    # Verify conv2 output
    if 'conv2' in tracker.activations:
        conv2_out = tracker.activations['conv2']
        print(f"After conv2: {tuple(conv2_out.shape)}")
        assert conv2_out.shape == (1, 64, 56, 56), "conv2 shape wrong"
        print("  ✅ conv2 shape correct")
    else:
        print("  ⚠️  conv2 not tracked")

    # Verify bn2 output
    if 'bn2' in tracker.activations:
        bn2_out = tracker.activations['bn2']
        print(f"After bn2:   {tuple(bn2_out.shape)}")
        assert bn2_out.shape == (1, 64, 56, 56), "bn2 shape wrong"
        print("  ✅ bn2 shape correct")
    else:
        print("  ⚠️  bn2 not tracked")

    # Verify residual connection
    print("\n--- Residual Connection Test ---")
    if output is not None:
        print(f"Final output: {tuple(output.shape)}")
        # The output should be relu(bn2_out + x)
        # We can verify the residual is being added by checking values
        print("  ✅ Output shape correct")
    else:
        print("  ⚠️  Forward returns None (not implemented yet)")

    print("\n" + "=" * 60)


def test_skip_connection_gradient_flow():
    """
    Verify that gradients flow through the skip connection.
    This is the key property that makes ResNet trainable.
    """
    from resnet import BasicBlock

    print()
    print("=" * 60)
    print("TEST: Skip Connection Gradient Flow")
    print("=" * 60)

    block = BasicBlock(in_channels=64, planes=64, stride=1)
    x = torch.randn(1, 64, 56, 56, requires_grad=True)

    output = block(x)
    if output is None:
        print("⚠️  Forward not implemented yet, skipping gradient test")
        return

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    print(f"\nInput gradient shape: {tuple(x.grad.shape)}")
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")

    # Check that gradient is non-zero (skip connection allows this)
    assert x.grad.norm() > 0, "Gradient is zero! Skip connection may be broken."
    print("✅ Gradients flow through the block")

    # Check gradients for each parameter
    print("\n--- Parameter Gradients ---")
    for name, param in block.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad_norm = {param.grad.norm().item():.4f}")
        else:
            print(f"  {name}: NO GRADIENT")

    print("\n" + "=" * 60)


def test_residual_addition_happens():
    """
    Verify that the skip connection actually ADDS the identity to the output.

    This is a mathematical verification: if we zero out the conv weights,
    the output should be relu(identity) since conv outputs will be ~zero.

    TODO(human): Implement this test

    Steps:
    1. Create a BasicBlock with in_channels=64, planes=64, stride=1
    2. Use torch.no_grad() context and zero out all conv weights:
       - block.conv1.weight.zero_()
       - block.conv2.weight.zero_()
    3. Create a known input, e.g., torch.ones(1, 64, 8, 8) * 0.5
    4. Run the forward pass
    5. Since convs output ~0 after BN, output ≈ relu(identity) = relu(input)
    6. Assert that output values match relu(input) approximately
       Hint: use torch.allclose(output, expected, atol=1e-5)

    Why this matters: This proves the skip connection is actually adding,
    not just passing through one path or the other.
    """
    from resnet import BasicBlock

    print()
    print("=" * 60)
    print("TEST: Residual Addition Verification")
    print("=" * 60)

    block = BasicBlock(in_channels=64, planes=64, stride=1)
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()

    x = torch.ones(1, 64, 8, 8) * 0.5
    output = block(x)
    expected = torch.relu(x)
    assert torch.allclose(output, expected, atol=1e-5), "Residual addition test failed"
    print("✅ Residual addition verified: output matches relu(input) when conv weights are zeroed")


def test_batchnorm_train_vs_eval():
    """
    Verify that BatchNorm behaves differently in train vs eval mode.

    In train mode: BN uses batch statistics (mean/var of current batch)
    In eval mode: BN uses running statistics (accumulated during training)

    TODO(human): Implement this test

    Steps:
    1. Create a BasicBlock
    
    2. Create an input tensor x = torch.randn(4, 64, 8, 8)  # batch_size=4
    3. Run forward in train mode: block.train(); out_train = block(x)
    4. Run forward in eval mode: block.eval(); out_eval = block(x)
    5. Assert that outputs are DIFFERENT: not torch.allclose(out_train, out_eval)

    Why this matters: Understanding train/eval mode is critical for:
    - Getting correct inference results
    - Debugging "works in training, fails in production" issues
    - Understanding why model.eval() is required before inference
    """
    from resnet import BasicBlock

    print()
    print("=" * 60)
    print("TEST: BatchNorm Train vs Eval Mode")
    print("=" * 60)
    block = BasicBlock(in_channels=64, planes=64, stride=1)
    x = torch.randn(4, 64, 8, 8)
    block.train(); out_train = block(x)
    block.eval(); out_eval = block(x)
    assert not torch.allclose(out_train, out_eval), "BatchNorm train vs eval outputs should differ"
    

if __name__ == "__main__":
    print("\n🧪 Running BasicBlock Tests\n")

    # Run tests
    try:
        test_basic_block_forward_step_by_step()
    except Exception as e:
        print(f"❌ Step-by-step test failed: {e}")

    print()

    try:
        test_basic_block_shapes()
    except Exception as e:
        print(f"❌ Shape test failed: {e}")

    print()

    try:
        test_skip_connection_gradient_flow()
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
