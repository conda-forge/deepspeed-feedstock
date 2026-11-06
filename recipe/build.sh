#!/bin/bash
set -ex

# Fix for https://github.com/conda-forge/deepspeed-feedstock/issues/1 to get pip_check
# working even without ninja as runtime dependency, xref
# https://github.com/conda-forge/causal-conv1d-feedstock/blob/bf0344b4740b6320723570071d9f7d6a2f5fd38e/recipe/meta.yaml#L20
sed -i.bak 's@ninja@#ninja@g' requirements/requirements.txt

if [[ ${cuda_compiler_version} != "None" ]]; then
  # There is no GPU on CI, so DeepSpeed would fall back to the CPU accelerator
  # and build only the CPU ops. Force the CUDA accelerator's op builders.
  export DS_ACCELERATOR=cuda
  export DS_BUILD_OPS=1
  # Need deepspeed-kernels (dskernels), which is not packaged on conda-forge
  export DS_BUILD_CUTLASS_OPS=0
  export DS_BUILD_RAGGED_DEVICE_OPS=0
  # Needs a CUTLASS source checkout
  export DS_BUILD_EVOFORMER_ATTN=0
  # Compile each extension's sources in parallel
  export DS_ENABLE_NINJA=1
  export MAX_JOBS=${CPU_COUNT}

  # Set the CUDA arch list from
  # https://github.com/conda-forge/pytorch-cpu-feedstock/blob/238fe50d9f9a3957584d3713531a81eec91e9f0e/recipe/build.sh#L217-L240
  # We could instead use CF_TORCH_CUDA_ARCH_LIST, available since CF pytorch 2.10?
  case ${cuda_compiler_version} in
      12.[89])
          export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX"
          ;;
      13.*)
          export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX"
          # c.f. https://github.com/pytorch/pytorch/pull/161316
          # export TORCH_NVCC_FLAGS="$TORCH_NVCC_FLAGS -compress-mode=size"
          ;;
      *)
          echo "No CUDA architecture list exists for CUDA v${cuda_compiler_version}. See build.sh for information on adding one."
          exit 1
  esac

else
  # Pre-compile the native CPU ops (cpu_adam, cpu_adagrad, cpu_lion,
  # fused_adam, deepspeed_shm_comm, async_io, pin_memory)
  export DS_ACCELERATOR=cpu
  export DS_BUILD_OPS=1
fi

# Skip `deepspeed_ccl_comm` (CPU accelerator): conda-forge's oneccl-devel only
# ships proprietary static libraries that need SYCL and Intel MPI; use torch's
# gloo + the SHM op for CPU comms instead.
export DS_BUILD_CCL_COMM=0

# The op builders default to -march=native and pick SIMD intrinsics from the
# build machine's CPU flags. Use portable settings instead (see patch 0001).
# Leave -march to the compiler flags from conda-forge's activation scripts.
export DS_CPU_ARCH=""
case ${target_platform} in
  linux-64)
    # Pick the SIMD code path from the x86_64 microarch level of this variant
    case ${microarch_level} in
      4)
        export DS_CPU_ARCH="-march=x86-64-v4"
        export DS_SIMD_WIDTH="-D__AVX512__"
        ;;
      3)
        export DS_CPU_ARCH="-march=x86-64-v3"
        export DS_SIMD_WIDTH="-D__AVX256__"
        ;;
      *)
        # conda-forge's x86_64 baseline has no AVX, so use the scalar code path
        export DS_SIMD_WIDTH="-D__SCALAR__"
        ;;
    esac
    ;;
  linux-aarch64|osx-arm64)
    # NEON is part of the AArch64 baseline
    export DS_SIMD_WIDTH="-D__NEON__"
    ;;
  *)
    export DS_SIMD_WIDTH="-D__SCALAR__"
    ;;
esac

# Disable sparse_attn since it requires an exact version of triton==1.0.0
export DS_BUILD_SPARSE_ATTN=0

${PYTHON} -m pip install . -vv
