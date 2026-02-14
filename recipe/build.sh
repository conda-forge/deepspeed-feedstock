#!/bin/bash
set -ex

# Fix for https://github.com/conda-forge/deepspeed-feedstock/issues/1 to get pip_check
# working even without ninja as runtime dependency, xref
# https://github.com/conda-forge/causal-conv1d-feedstock/blob/bf0344b4740b6320723570071d9f7d6a2f5fd38e/recipe/meta.yaml#L20
sed -i.bak 's@ninja@#ninja@g' requirements/requirements.txt

# Deepspeed ops cannot be built without CUDA
if [[ ${cuda_compiler_version} != "None" ]]; then
  export DS_BUILD_OPS=1

  # Set the CUDA arch list from
  # https://github.com/conda-forge/pytorch-cpu-feedstock/blob/238fe50d9f9a3957584d3713531a81eec91e9f0e/recipe/build.sh#L217-L240
  # We could instead use CF_TORCH_CUDA_ARCH_LIST, available since CF pytorch 2.10?
  case ${cuda_compiler_version} in
      12.[89])
          export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX"
          ;;
      13.0)
          export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX"
          # c.f. https://github.com/pytorch/pytorch/pull/161316
          # export TORCH_NVCC_FLAGS="$TORCH_NVCC_FLAGS -compress-mode=size"
          ;;
      *)
          echo "No CUDA architecture list exists for CUDA v${cuda_compiler_version}. See build.sh for information on adding one."
          exit 1
  esac

else
  export DS_BUILD_OPS=0
fi

# Disable sparse_attn since it requires an exact version of triton==1.0.0
export DS_BUILD_SPARSE_ATTN=0

${PYTHON} -m pip install . -vv
