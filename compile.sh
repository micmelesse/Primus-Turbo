#!/bin/bash
# export PYTORCH_ROCM_ARCH=gfx942:xnack-
# export TORCH_DONT_CHECK_COMPILER_ABI=1
# export CFLAGS="-O3 -fPIC"
# # export CXXFLAGS="-O3 -fPIC --offload-arch=gfx942:xnack-"
# export CXXFLAGS="-O3 -fPIC"
# export HIP_CXX_FLAGS="-O3 -fPIC"
# export MPI_HOME=/install/ompi
# export UCX_HOME=/install/ucx
# export ROCSHMEM_HOME=/workspace/Primus-Turbo/install/rocshmem

pip uninstall primus_turbo -y
rm -rf primus_turbo/lib
rm -f primus_turbo/pytorch/_C*
pip install -r requirements.txt
python setup.py develop
