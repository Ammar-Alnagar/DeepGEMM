# GPU Architecture Support Matrix and Porting Notes

## Overview

DeepGEMM has been extended to support NVIDIA **SM80 (Ampere)** and **SM89 (Ada Lovelace)** architectures while fully preserving the existing **SM90 (Hopper)** and **SM100 (Blackwell)** implementations.

---

## Architecture Support Matrix

| Operation / Feature | SM80 (Ampere) | SM89 (Ada Lovelace) | SM90 (Hopper) | SM100 (Blackwell) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense BF16 GEMM (`bf16_gemm_*`)** | cuBLASLt Fallback | cuBLASLt Fallback | Native SM90 Kernel / cuBLASLt | Native SM100 Kernel / cuBLASLt |
| **Dense FP8 GEMM (`fp8_gemm_*`)** | Unsupported (Hardware Limit) | SM89 FP8 Kernel (`sm89_fp8_gemm_1d1d`) | Native SM90 Kernel (1D1D / 1D2D) | Native SM100 Kernel |
| **Microscaled FP4 GEMM (`fp4_gemm_*`)** | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Native SM100 FP4 Kernel |
| **Grouped BF16 GEMM (`m_grouped_bf16_*`)** | Unsupported | Unsupported | Native SM90 Kernel | Native SM100 Kernel |
| **Grouped FP8 GEMM (`m_grouped_fp8_*`)** | Unsupported (Hardware Limit) | SM89 Grouped FP8 Kernel | Native SM90 Kernel | Native SM100 Kernel |
| **V3.2 MQA Logits (`fp8_fp4_mqa_logits`)** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |
| **Mega MoE (`fp8_fp4_mega_moe`)** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |
| **HyperConnection & Mega Gate / mHC** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |

---

## Architecture Details & Error Handling

1. **SM80 (Ampere, compute capability 8.0)**:
   - Does not have native hardware FP8 Tensor Cores or Blackwell FP4 hardware.
   - Standard dense BF16 GEMMs automatically dispatch through cuBLASLt.
   - Native FP8 operations invoked on SM80 raise an explicit `RuntimeError` stating: `FP8 Tensor Core GEMM is not natively supported on NVIDIA Ampere (SM80) due to hardware limits. Use BF16 GEMM instead.`

2. **SM89 (Ada Lovelace, compute capability 8.9)**:
   - Natively supports FP8 Tensor Cores (`sm89_fp8_gemm_1d1d`).
   - Standard dense BF16 GEMMs dispatch through cuBLASLt.
   - Blackwell FP4 calls raise a clear `RuntimeError` indicating hardware requirements.

3. **SM90 (Hopper) & SM100 (Blackwell)**:
   - Retain full native performance, TMA/WGMMA pipelines, UE8M0 scaling, and custom epilogue support without any changes or regressions.

---

## Running Validation Tests on GPU

A dedicated physical hardware test suite is provided in `tests/test_sm80_sm89.py`.

To run tests when cloned onto a system with an SM80 or SM89 GPU:

```bash
python3 -m unittest tests/test_sm80_sm89.py
```

This test checks:
1. GPU Compute Capability detection.
2. Correctness of BF16 GEMM execution via cuBLASLt fallback against PyTorch references.
3. FP8 GEMM execution on SM89 Ada Lovelace GPUs.
4. Proper handling and clear runtime architecture error messages for unsupported low-precision hardware features on SM80.
