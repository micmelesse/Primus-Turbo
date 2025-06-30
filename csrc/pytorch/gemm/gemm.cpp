#include <hipblaslt/hipblaslt.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

static hipDataType get_hipblaslt_dtype(const at::ScalarType t) {
    switch (t) {
    case at::kHalf:
        return HIP_R_16F;
    case at::kFloat:
        return HIP_R_32F;
    case at::kBFloat16:
        return HIP_R_16BF;
    case at::kFloat8_e4m3fnuz:
        return HIP_R_8F_E4M3_FNUZ;
    case at::kFloat8_e4m3fn:
        return HIP_R_8F_E4M3;
    case at::kFloat8_e5m2fnuz:
        return HIP_R_8F_E5M2_FNUZ;
    case at::kFloat8_e5m2:
        return HIP_R_8F_E5M2;
    default:
        AT_ERROR("Invalid type");
    }
}

static inline bool is_16bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

static inline bool is_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

at::Tensor gemm(const at::Tensor A, const at::Tensor B, const at::ScalarType out_dtype,
                const bool transA, const bool transB) {
    PRIMUS_CHECK(is_16bit_floating_point_dtype(A.scalar_type()));
    PRIMUS_CHECK(is_16bit_floating_point_dtype(B.scalar_type()));
    PRIMUS_CHECK(A.scalar_type() == B.scalar_type(), "A and B dtype mismatch");
    PRIMUS_CHECK(is_floating_point_dtype(out_dtype));

    PRIMUS_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_CHECK(B.is_contiguous(), "B must be contiguous");

    PRIMUS_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    const int64_t m = transB ? B.size(0) : B.size(1);
    const int64_t k = transB ? B.size(1) : B.size(0);
    const int64_t n = transA ? A.size(1) : A.size(0);

    int64_t lda, ldb, ldd;
    if (!transA && transB) { // NT
        PRIMUS_CHECK(A.size(1) == B.size(1), "tensor size mismatch");
        lda = k;
        ldb = k;
        ldd = m;
    } else if (!transA && !transB) { // NN
        PRIMUS_CHECK(A.size(1) == B.size(0), "tensor size mismatch");
        lda = m;
        ldb = k;
        ldd = m;
    } else if (transA && !transB) { // TN
        PRIMUS_CHECK(A.size(0) == B.size(0), "tensor size mismatch");
        lda = m;
        ldb = n;
        ldd = m;
    } else {
        PRIMUS_ERROR("Not support layout.");
    }

    at::Tensor C = at::empty({n, m}, torch::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::hip::getCurrentHIPStream();
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();

    hipblasOperation_t trans_operation_A = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_operation_B = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipDataType  A_type            = get_hipblaslt_dtype(A.scalar_type());
    const hipDataType  B_type            = get_hipblaslt_dtype(B.scalar_type());
    const hipDataType  C_type            = get_hipblaslt_dtype(C.scalar_type());
    const hipDataType  D_type            = C_type;

    hipblasLtMatmulDesc_t       operation_desc = nullptr;
    hipblasLtMatrixLayout_t     A_desc = nullptr, B_desc = nullptr, D_desc = nullptr;
    hipblasLtMatmulPreference_t preference        = nullptr;
    hipblasLtEpilogue_t         epilogue          = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasComputeType_t        gemm_compute_type = HIPBLAS_COMPUTE_32F;

    // TODO(ruibzhan): workspace size of gfx950 is 64MiB.
    const int64_t workspace_size = 33554432;
    at::Tensor workspace = at::empty({workspace_size}, torch::dtype(at::kByte).device(at::kCUDA));

    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatrixLayoutCreate(&A_desc, A_type, trans_operation_A == HIPBLAS_OP_N ? k : n,
                                    trans_operation_A == HIPBLAS_OP_N ? n : k, ldb));
    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatrixLayoutCreate(&B_desc, B_type, trans_operation_B == HIPBLAS_OP_N ? m : k,
                                    trans_operation_B == HIPBLAS_OP_N ? k : m, lda));
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&D_desc, D_type, m, n, ldd));

    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatmulDescCreate(&operation_desc, gemm_compute_type, HIP_R_32F));
    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatmulDescSetAttribute(operation_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &trans_operation_B, sizeof(trans_operation_B)));
    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatmulDescSetAttribute(operation_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &trans_operation_A, sizeof(trans_operation_A)));
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    int                                           algo_count = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    algos.resize(algo_count);

    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    PRIMUS_CHECK_HIPBLASLT(
        hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_size, sizeof(workspace_size)));

    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle, operation_desc, B_desc, A_desc,
                                                           D_desc, D_desc, preference, algo_count,
                                                           algos.data(), &algo_count));

    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceDestroy(preference));

    const float one  = 1.0;
    const float zero = 0.0;

    // NOTE: hipblaslt expects tensor in col-major but torch Tensor is in row-major.
    // Swapping A&B that are essentially computing C^T = B^T @ A^T.
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmul(
        handle, operation_desc, static_cast<const void *>(&one),
        static_cast<const void *>(B.data_ptr()), B_desc, static_cast<const void *>(A.data_ptr()),
        A_desc, static_cast<const void *>(&zero), static_cast<const void *>(C.data_ptr()), D_desc,
        static_cast<void *>(C.data_ptr()), D_desc, &algos[0].algo,
        static_cast<void *>(workspace.data_ptr()), workspace_size, stream));

    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(D_desc));
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(B_desc));
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(A_desc));
    PRIMUS_CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(operation_desc));

    return C;
}

} // namespace primus_turbo::pytorch
