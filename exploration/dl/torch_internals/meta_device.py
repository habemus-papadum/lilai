import torch
from torchvision.models import resnet18

def bytes_of(t: torch.Tensor) -> int:
    # Estimated storage if this tensor were materialized on a real device
    return t.numel() * t.element_size()

def format_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"

# 1) Build the model directly on meta (avoids allocating real parameter storage)
#    (Torchvision's default is weights=None, but we pass it explicitly.)
with torch.device("meta"):
    model = resnet18(weights=None)
model.eval()

# 2) Dummy input on meta
x = torch.empty(1, 3, 224, 224, device="meta", dtype=torch.float32)

# 3) Collect activation metadata
records = []

def make_hook(name: str):
    def hook(module, inputs, output):
        # output can be Tensor, tuple/list, dict; handle common cases
        def add(o, suffix=""):
            if torch.is_tensor(o):
                records.append({
                    "name": f"{name}{suffix}",
                    "shape": tuple(o.shape),
                    "dtype": str(o.dtype).replace("torch.", ""),
                    "numel": o.numel(),
                    "bytes": bytes_of(o),
                })
            elif isinstance(o, (tuple, list)):
                for i, oi in enumerate(o):
                    add(oi, suffix=f"[{i}]")
            elif isinstance(o, dict):
                for k, oi in o.items():
                    add(oi, suffix=f"[{k!r}]")
        add(output)
    return hook

hooks = []
for name, m in model.named_modules():
    if name == "":
        continue

    # If you want *everything*, remove this "leaf module" filter.
    # Leaf modules gives a clean “layer-like” list (Conv/BN/ReLU/Pool/etc).
    is_leaf = len(list(m.children())) == 0
    if is_leaf:
        hooks.append(m.register_forward_hook(make_hook(name)))

# 4) Run one meta forward (no real compute/storage)
with torch.no_grad():
    _ = model(x)

# 5) Cleanup hooks
for h in hooks:
    h.remove()

# 6) Print activation sizes
total = 0
for r in records:
    total += r["bytes"]
    print(f"{r['name']:<35} shape={r['shape']!s:<18} dtype={r['dtype']:<8} est={format_bytes(r['bytes'])}")

print(f"\nSum of recorded activations (not peak): {format_bytes(total)}")
