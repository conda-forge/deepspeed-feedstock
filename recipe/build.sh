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
  # https://github.com/conda-forge/pytorch-cpu-feedstock/blob/c5ded360dcc4f62d6cd98b3748c2a10c50aa45f7/recipe/build.sh#L220
  if [[ ${cuda_compiler_version} == 12.* ]]; then
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
  else
    echo "Unsupported cuda version. edit build.sh"
    exit 1
  fi

else
  export DS_BUILD_OPS=0
fi

# Disable sparse_attn since it requires an exact version of triton==1.0.0
export DS_BUILD_SPARSE_ATTN=0

${PYTHON} -m pip install . -vv
