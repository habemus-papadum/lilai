import torch
import torch.fx as fx
from torchvision.models import resnet18

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp






# Doesn't work




# Build model on meta to avoid parameter storage (optional but nice for huge models)
with torch.device("meta"):
    model = resnet18(weights=None).eval()

gm = fx.symbolic_trace(model)

# Create a fake tensor mode. Inside this context, tensors are "fake" (no data),
# but can pretend to be on a real device (e.g. "cuda") for more accurate metadata.
fake_mode = FakeTensorMode()
with fake_mode:
    fake_x = torch.empty(1, 3, 224, 224, device="cuda", dtype=torch.float32)

# Propagate fake tensors through the FX graph. This fills node.meta["val"].
FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

for node in gm.graph.nodes:
    v = node.meta.get("val", None)
    if isinstance(v, torch.Tensor):
        est_bytes = v.numel() * v.element_size()
        print(
            f"{node.op:<12} {str(node.target):<40} "
            f"shape={tuple(v.shape)!s:<18} dtype={v.dtype} device={v.device} "
            f"est_bytes={est_bytes}"
        )
