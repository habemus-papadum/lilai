"""
Exploring PyTorch's nn.Module magic for detecting submodules and parameters.
Run this file to see how the registries work!
"""

import torch
import torch.nn as nn


class DebugModule(nn.Module):
    """
    A module that prints when __setattr__ is called,
    so we can see what's happening behind the scenes.
    """

    def __setattr__(self, name: str, value) -> None:
        # Show what's being set
        type_name = type(value).__name__

        if isinstance(value, nn.Parameter):
            print(f"  📦 Setting PARAMETER: {name!r} ({type_name})")
        elif isinstance(value, nn.Module):
            print(f"  🧩 Setting MODULE: {name!r} ({type_name})")
        elif isinstance(value, torch.Tensor):
            print(f"  📊 Setting TENSOR (not registered!): {name!r}")
        else:
            print(f"  📝 Setting ATTRIBUTE: {name!r} ({type_name})")

        # Call parent to actually do the registration
        super().__setattr__(name, value)


class MyModel(DebugModule):
    def __init__(self):
        print("\n1. Calling super().__init__():")
        super().__init__()

        print("\n2. Setting a submodule (Conv2d):")
        self.conv = nn.Conv2d(3, 64, 3)

        print("\n3. Setting a Parameter:")
        self.my_scale = nn.Parameter(torch.ones(64))

        print("\n4. Setting a raw tensor (WON'T be tracked!):")
        self.raw_tensor = torch.zeros(10)

        print("\n5. Setting a regular Python attribute:")
        self.name = "my_model"

        print("\n6. Registering a buffer (explicit method):")
        self.register_buffer("running_mean", torch.zeros(64))


def main():
    print("=" * 60)
    print("Creating MyModel - watch __setattr__ intercept everything!")
    print("=" * 60)

    model = MyModel()

    print("\n" + "=" * 60)
    print("Inspecting the internal registries:")
    print("=" * 60)

    print("\n📦 _parameters registry:")
    for name, param in model._parameters.items():
        print(f"  {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")

    print("\n🧩 _modules registry:")
    for name, module in model._modules.items():
        print(f"  {name}: {module.__class__.__name__}")

    print("\n📊 _buffers registry:")
    for name, buf in model._buffers.items():
        print(f"  {name}: shape={tuple(buf.shape)}, requires_grad={buf.requires_grad}")

    print("\n" + "=" * 60)
    print("What .parameters() returns (recursive):")
    print("=" * 60)
    for name, param in model.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")

    print("\n⚠️  Notice: raw_tensor is NOT in any registry!")
    print(f"  model.raw_tensor exists: {hasattr(model, 'raw_tensor')}")
    print(f"  But it won't be saved/loaded or moved to GPU with .to()")

    print("\n" + "=" * 60)
    print("state_dict() - what gets saved:")
    print("=" * 60)
    for key in model.state_dict().keys():
        print(f"  {key}")
    print("\n  (raw_tensor is missing! Use register_buffer for persistent tensors)")


if __name__ == "__main__":
    main()
