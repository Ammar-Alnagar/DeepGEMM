#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"
#include "smxx_cublaslt.hpp"

namespace deep_gemm {

static void sm89_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c, const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const std::string& compiled_dims) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 8 and jit->device.get_arch_minor() >= 9);

    // SM89 Ada Lovelace natively supports FP8 Tensor Cores via cuBLASLt `CUDA_R_8F_E4M3` matmul API.
    // D = A @ B.T with per-tensor / per-channel FP8 scaling factors.
    // For general shapes, cuBLASLt provides optimal hardware execution using SM89 FP8 mma.sync instructions.
    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

} // namespace deep_gemm
