#include "grouped_gemm.h"
#include "grouped_gemm.hpp"
namespace primus_turbo {
void invoke_gemm(int group_count, const std::vector<grouped_gemm_kargs> &args, hipStream_t stream) {
    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));
    float ave_time = 0;
    // Regular version of grouped gemm
    grouped_gemm<ADataType, BDataType, DsDataType, AccDataType, CDataType, ALayout, BLayout,
                 DsLayout, CLayout, CDEElementWise>(args, ck_tile::stream_config{stream},
                                                    gemm_workspace.GetDeviceBuffer());
}

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout>
void ck_grouped_gemm_kernel(const ADataType *a_ptr, // a_ptr b_ptr c_ptr from gpu src
                            const BDataType *b_ptr, CDataType *c_ptr,
                            const int *seg_lens, // seg_lens from gpu src
                            const int B, const int N, const int K, hipStream_t stream) {
    int                           group_count = B;
    std::vector<ck_tile::index_t> stride_As;
    std::vector<ck_tile::index_t> stride_Bs;
    std::vector<ck_tile::index_t> stride_Cs;
    stride_As.reserve(group_count);
    stride_Bs.reserve(group_count);
    stride_Cs.reserve(group_count);

    ALayout a_layout{};
    BLayout b_layout{};
    CLayout c_layout{};

    // Initialize strides based on the input segment lengths
    for (int i = 0; i < group_count; i++) {
        stride_As.push_back(K);
        stride_Bs.push_back(K);
        stride_Cs.push_back(N);
        stride_As[i] =
            ck_tile::get_default_stride(seg_lens[i], K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] =
            ck_tile::get_default_stride(seg_lens[i], N, stride_Cs[i], is_row_major(c_layout));
    }
    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);
    // printf("%d ",seg_lens[0]);
    // printf("%d \n",seg_lens[1]);
    gemm_descs.push_back({a_ptr,
                          b_ptr,
                          {},
                          c_ptr,
                          1,
                          seg_lens[0],
                          N,
                          K,
                          stride_As[0],
                          stride_Bs[0],
                          {},
                          stride_Cs[0]});
    // for(int j = 0; j < 100; ++j)
    // {
    //     printf("0 %f %f %f \n ", static_cast<float>(a_ptr[j+511*K]),
    //     static_cast<float>(b_ptr[j+2047*N]), static_cast<float>(c_ptr[j+511*N]));
    // }
    for (int i = 1; i < group_count; ++i) {

        a_ptr += seg_lens[i - 1] * K;
        b_ptr += N * K;
        c_ptr += N * seg_lens[i - 1];

        gemm_descs.push_back({a_ptr,
                              b_ptr,
                              {},
                              c_ptr,
                              1,
                              seg_lens[i],
                              N,
                              K,
                              stride_As[i],
                              stride_Bs[i],
                              {},
                              stride_Cs[i]});
    }
    invoke_gemm<ADataType, BDataType, ck_tile::tuple<>, AccDataType, CDataType, ALayout, BLayout,
                ck_tile::tuple<>, CLayout, false>(group_count, gemm_descs, stream);
}
} // namespace primus_turbo
