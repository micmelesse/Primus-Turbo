#include "ck_tile/host/hip_check_error.hpp"
#include "grouped_gemm.hpp"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

void ck_test() {
    printf("ck_test\n");
}

template <typename Layout> static constexpr inline auto is_row_major(Layout layout_) {
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename CDataType>
void ck_grouped_gemm_kernel(const ADataType *p_a, // p_a p_b p_c from gpu src
                            const BDataType *p_b, CDataType *p_c,
                            const int *p_seg_lens, // p_seg_lens from gpu src
                            const int B, const int N, const int K) {
    // void ck_grouped_gemm_kernel(const ck_tile::half_t *p_a, // p_a p_b p_c from gpu src
    //                             const ck_tile::half_t *p_b, ck_tile::half_t *p_c,
    //                             const int *p_seg_lens, // p_seg_lens from gpu src
    //                             const int B, const int N, const int K) {
    printf("ck_grouped_gemm_kernel\n");
    // using AccDataType = float;

    // // Create gemm descriptors for grouped gemm
    // std::vector<grouped_gemm_kargs> gemm_descs;
    // gemm_descs.reserve(B);

    // // Initialize strides - using default values for simplicity
    // // In a more complete implementation, these might be passed as parameters
    // const ck_tile::index_t stride_A = K;
    // const ck_tile::index_t stride_B = N;
    // const ck_tile::index_t stride_C = N;

    // // Create descriptors for each group
    // const ADataType *cur_a = p_a;
    // const BDataType *cur_b = p_b;
    // CDataType       *cur_c = p_c;

    // for (int i = 0; i < B; i++) {
    //     const int M = p_seg_lens[i];

    //     gemm_descs.push_back({
    //         cur_a,    // a_ptr
    //         cur_b,    // b_ptr
    //         {},       // ds_ptr (empty for no D tensors)
    //         cur_c,    // e_ptr/c_ptr
    //         1,        // k_batch
    //         M,        // M
    //         N,        // N
    //         K,        // K
    //         stride_A, // stride_A
    //         stride_B, // stride_B
    //         {},       // stride_Ds (empty)
    //         stride_C  // stride_E/stride_C
    //     });

    //     // Move pointers to next group
    //     cur_a += M * K;
    //     cur_b += K * N; // B matrix is shared across groups
    //     cur_c += M * N;
    // }

    // // Allocate workspace for kernel arguments
    // ck_tile::DeviceMem gemm_workspace;
    // gemm_workspace.Realloc(get_workspace_size(gemm_descs));

    // // Prepare kernel arguments
    // std::vector<ck_tile::GemmTransKernelArg> kargs;
    // void                                    *kargs_ptr = gemm_workspace.GetDeviceBuffer();

    // for (const auto &arg : gemm_descs) {
    //     kargs.emplace_back(ck_tile::GemmKernelArgs<>{arg.a_ptr,
    //                                                  arg.b_ptr,
    //                                                  {},
    //                                                  arg.e_ptr,
    //                                                  arg.M,
    //                                                  arg.N,
    //                                                  arg.K,
    //                                                  arg.stride_A,
    //                                                  arg.stride_B,
    //                                                  {},
    //                                                  arg.stride_E,
    //                                                  arg.k_batch});
    // }

    // // Copy kernel arguments to device
    // const auto stream =
    //     ck_tile::stream_config{nullptr, true, 1, 20, 100}; // No need to time in this function
    // HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr, kargs.data(),
    //                                     kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
    //                                     hipMemcpyHostToDevice, stream.stream_id_));

    // // Execute the grouped GEMM operation
    // float ave_time =
    //     grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout,
    //                           CLayout>(stream, B, kargs_ptr, false /* splitk */);

    // std::string op_name{"Grouped Gemm"};
    // std::size_t flop = 0, num_btype = 0;
    // for (int j = 0; j < B; ++j) {
    //     flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;
    //     num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
    //                  sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
    //                  sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N;
    // }
    // float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    // float gb_per_sec = num_btype / 1.E6 / ave_time;
    // std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
    //           << gb_per_sec << " GB/s, " << op_name << std::endl;
}

} // namespace primus_turbo
