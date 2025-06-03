#!/bin/bash

# Add CUDA paths (already present)
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64

# Add OpenCL paths
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# If using a vendor SDK (uncomment and modify accordingly)
# export OpenCL_SDK=/path/to/opencl/sdk
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:$OpenCL_SDK/include
# export LIBRARY_PATH=$LIBRARY_PATH:$OpenCL_SDK/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OpenCL_SDK/lib

# Compile in each subdirectory
for currDir in *; do
    echo $currDir
    if [ -d "$currDir" ]; then
        cd "$currDir"
        pwd
        make clean
        make
        cd ..
    fi
done