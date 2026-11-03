import shutil
import sys

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.git_version_info import installed_ops, torch_info

assert get_accelerator()._name == "cpu", get_accelerator()._name
# Make sure nothing below can silently fall back to a JIT build
assert shutil.which("ninja") is None, "ninja must not be available for this test"

print("installed_ops:", installed_ops)
print("torch_info:", torch_info)
assert torch_info["version"] != "0.0", torch_info

expected = ["cpu_adam", "cpu_adagrad", "cpu_lion", "fused_adam"]
if sys.platform == "linux":
    expected += ["deepspeed_shm_comm", "async_io", "pin_memory"]

missing = [op for op in expected if not installed_ops.get(op)]
assert not missing, f"ops not pre-compiled: {missing}"

from deepspeed.ops.op_builder.all_ops import ALL_OPS  # noqa: E402

for op in expected:
    ALL_OPS[op].load()
    print(f"loaded {op}")

# Exercise the CPU Adam kernel
from deepspeed.ops.adam import DeepSpeedCPUAdam  # noqa: E402

torch.manual_seed(0)
p = torch.nn.Parameter(torch.randn(1024))
p_ref = torch.nn.Parameter(p.detach().clone())
opt = DeepSpeedCPUAdam([p], lr=1e-2)
opt_ref = torch.optim.AdamW([p_ref], lr=1e-2, weight_decay=0.0)
for _ in range(5):
    g = torch.randn(1024)
    p.grad = g.clone()
    p_ref.grad = g.clone()
    opt.step()
    opt_ref.step()
torch.testing.assert_close(p, p_ref, rtol=1e-4, atol=1e-5)
print("DeepSpeedCPUAdam matches torch.optim.AdamW")
