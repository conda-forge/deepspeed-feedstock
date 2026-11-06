"""Check that DeepSpeed's ops were pre-compiled and load without a JIT build.

Usage: python test_ops.py {cpu,cuda}
"""
import os
import sys

accelerator = sys.argv[1]
# The pre-compiled ops are only used when the runtime accelerator matches the
# one they were built for. Without a GPU (as on CI) DeepSpeed would pick "cpu",
# and on Apple Silicon it would pick "mps", so set it explicitly.
os.environ["DS_ACCELERATOR"] = accelerator

import torch  # noqa: E402

from deepspeed.accelerator import get_accelerator  # noqa: E402
from deepspeed.git_version_info import accelerator_name, installed_ops, torch_info  # noqa: E402
from deepspeed.ops.op_builder.all_ops import ALL_OPS  # noqa: E402

assert get_accelerator()._name == accelerator, get_accelerator()._name
assert accelerator_name == accelerator, f"package was built for {accelerator_name}"
# There is no compiler in the test env, so any op that was not pre-compiled
# fails to JIT build when loaded below instead of silently passing.

print("installed_ops:", installed_ops)
print("torch_info:", torch_info)
assert torch_info["version"] != "0.0", torch_info

if accelerator == "cpu":
    expected = ["cpu_adam", "cpu_adagrad", "cpu_lion", "fused_adam"]
    if sys.platform == "linux":
        expected += ["deepspeed_shm_comm", "async_io", "pin_memory"]
else:
    # Keep in sync with the DS_BUILD_*=0 settings in build.sh
    disabled = {"cutlass_ops", "ragged_device_ops", "evoformer_attn"}
    assert torch_info["cuda_version"] != "0.0", torch_info
    expected = sorted(set(ALL_OPS) - disabled)

missing = [op for op in expected if not installed_ops.get(op)]
assert not missing, f"ops not pre-compiled: {missing}"

needs_driver = set()
if accelerator == "cuda":
    import ctypes

    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        # No NVIDIA driver (e.g. CI): ops linking libcuda.so.1 cannot be loaded
        needs_driver = {"gds"}

for op in expected:
    if op in needs_driver:
        print(f"skipped loading {op}: libcuda.so.1 not available")
        continue
    ALL_OPS[op].load()
    print(f"loaded {op}")

if accelerator == "cpu":
    # Exercise the CPU Adam kernel
    from deepspeed.ops.adam import DeepSpeedCPUAdam

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
