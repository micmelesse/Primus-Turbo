// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "../extensions.h"
#include <vector>

namespace primus_turbo::pytorch {

// This is a helper function to provide a Python interface to create a HIP stream with CU masks.
// The function returns the stream pointer cast as an int64_t to be compatible with Python's int type.
// Example of usage in Python:
//     cu_masks = [0x0000000F, 0x00001234]  # Example CU masks for a 2-CU GPU
//     stream_ptr = create_stream_with_cu_masks(cu_masks)
//     stream = torch.cuda.get_stream_from_external(stream_ptr)
//     with torch.cuda.stream(stream):
//         # Your CUDA operations here
// Note: The caller is responsible for destroying the created stream.
int64_t create_stream_with_cu_masks(const std::vector<uint32_t>& cu_masks) {
    hipStream_t hip_stream;
    hipError_t err = hipExtStreamCreateWithCUMask(&hip_stream, cu_masks.size(), cu_masks.data());
    if (err != hipSuccess) {
        std::cerr << "Error creating stream with CU mask: " << hipGetErrorString(err) << std::endl;
        // return null pointer
        return 0;
    }
    return reinterpret_cast<int64_t>(hip_stream);
}

// This is a helper function to provide a Python interface to destroy a HIP stream.
void destroy_stream(const int64_t stream_ptr) {
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream_ptr);
    hipError_t err = hipStreamDestroy(hip_stream);
    if (err != hipSuccess) {
        std::cerr << "Error destroying stream: " << hipGetErrorString(err) << std::endl;
    }
}

} // namespace primus_turbo::pytorch
