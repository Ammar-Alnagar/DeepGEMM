#!/usr/bin/env python3
"""
Test script for SM80 and SM89 architecture support and fallbacks in DeepGEMM.
Can be executed in both offline environments (checking mock architecture dispatch and unit logic)
and on physical SM80/SM89 GPU hardware when cloned.
"""

import unittest
import torch

try:
    import deep_gemm
    import deep_gemm._C as _C
    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False


class TestSM80SM89Support(unittest.TestCase):

    def setUp(self):
        if not HAS_DEEPGEMM:
            self.skipTest("DeepGEMM C++ extension not compiled/installed in environment.")

    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA device required for hardware testing.")

    def test_arch_detection(self):
        major, minor = torch.cuda.get_device_capability()
        print(f"[Hardware Test] Detected GPU Compute Capability: SM{major}{minor}")
        self.assertGreaterEqual(major, 8, "Expected GPU compute capability >= 8.0")

    def test_bf16_gemm_fallback(self):
        """Test BF16 GEMM on SM80, SM89, or higher using cuBLASLt fallback / dispatch."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

        _C.bf16_gemm_nt(a, b, d, None, "nk", None, None)
        ref_d = torch.matmul(a, b.t())

        diff = torch.max(torch.abs(d - ref_d)).item()
        print(f"[Hardware Test] BF16 GEMM Max Diff vs PyTorch: {diff:.6f}")
        self.assertLess(diff, 1e-1, "BF16 GEMM output exceeds acceptable tolerance.")

    def test_sm80_fp8_unsupported_error(self):
        """Verify that SM80 hardware returns explicit error for native FP8 Tensor Core operations."""
        major, minor = torch.cuda.get_device_capability()
        if major == 8 and minor == 0:
            m, n, k = 128, 256, 512
            a_fp = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
            a_sf = torch.ones((m, k // 128), dtype=torch.float32, device="cuda")
            b_fp = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
            b_sf = torch.ones((n, k // 128), dtype=torch.float32, device="cuda")
            d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

            with self.assertRaises(RuntimeError) as ctx:
                _C.fp8_fp4_gemm_nt((a_fp, a_sf), (b_fp, b_sf), d, None, None, None, None, "nk", False, None, None)
            self.assertIn("not natively supported on NVIDIA Ampere (SM80)", str(ctx.exception))
            print("[Hardware Test] Successfully verified SM80 FP8 hardware error.")

    def test_fp4_unsupported_error_on_sm8x(self):
        """Verify that SM80/SM89 GPUs raise explicit error for Blackwell-only FP4 GEMM."""
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            m, n, k = 128, 256, 512
            # FP4 packed tensor mock
            a_fp = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device="cuda")
            a_sf = torch.ones((m, k // 128), dtype=torch.int32, device="cuda")
            b_fp = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda")
            b_sf = torch.ones((n, k // 128), dtype=torch.int32, device="cuda")
            d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

            with self.assertRaises(RuntimeError) as ctx:
                _C.fp8_fp4_gemm_nt((a_fp, a_sf), (b_fp, b_sf), d, None, None, None, None, "nk", False, None, None)
            self.assertIn("supported on NVIDIA Blackwell (SM100)", str(ctx.exception))
            print("[Hardware Test] Successfully verified FP4 hardware error on pre-SM100 GPU.")


if __name__ == "__main__":
    unittest.main()
