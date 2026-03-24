// AMD MFMA Matrix Core Programming Examples
// Source: Adapted from ROCm Matrix Core Programming Guide (2025-09-30)
// These examples demonstrate MFMA compiler intrinsics for various precisions.
// Each kernel is launched with a single wavefront (64 threads).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// =============================================================================
// Example 1: mfma_f32_32x32x2_f32 — FP32 input, FP32 output
// =============================================================================
// Each of 64 threads holds:
//   A: 1 FP32 element
//   B: 1 FP32 element
//   C/D: 16 FP32 elements (accumulator)

__global__ void mfma_f32_32x32x2f32_kernel(const float* A, const float* B,
                                             float* C, float* D) {
    int tid = threadIdx.x;  // 0..63

    // Load A: thread tid holds A[row][col] where row = tid % 32, col = tid / 32
    float a = A[tid];

    // Load B: thread tid holds B[row][col] where row = tid / 32, col = tid % 32
    float b = B[tid];

    // Load C accumulator: each thread holds 16 elements
    float c[16];
    for (int i = 0; i < 16; i++)
        c[i] = C[tid * 16 + i];

    // Execute MFMA: D = A * B + C (32x32x2 tile)
    // cbsz=0, abid=0, blgp=0 for standard GEMM
    *reinterpret_cast<__attribute__((ext_vector_type(16))) float*>(c) =
        __builtin_amdgcn_mfma_f32_32x32x2f32(a, b,
            *reinterpret_cast<__attribute__((ext_vector_type(16))) float*>(c),
            0, 0, 0);

    // Store D: each thread writes 16 elements
    for (int i = 0; i < 16; i++)
        D[tid * 16 + i] = c[i];
}


// =============================================================================
// Example 2: mfma_f32_16x16x16_f16 — FP16 input, FP32 output
// =============================================================================
// Each of 64 threads holds:
//   A: 4 FP16 elements
//   B: 4 FP16 elements
//   C/D: 4 FP32 elements (accumulator)

__global__ void mfma_f32_16x16x16f16_kernel(const _Float16* A, const _Float16* B,
                                              float* C, float* D) {
    int tid = threadIdx.x;

    // Load A: each thread holds 4 FP16 values
    typedef _Float16 half4 __attribute__((ext_vector_type(4)));
    half4 a;
    for (int i = 0; i < 4; i++)
        a[i] = A[tid * 4 + i];

    // Load B: each thread holds 4 FP16 values
    half4 b;
    for (int i = 0; i < 4; i++)
        b[i] = B[tid * 4 + i];

    // Load C accumulator: each thread holds 4 FP32 values
    typedef float float4_mfma __attribute__((ext_vector_type(4)));
    float4_mfma c;
    for (int i = 0; i < 4; i++)
        c[i] = C[tid * 4 + i];

    // Execute MFMA: 16x16x16 tile
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);

    // Store results
    for (int i = 0; i < 4; i++)
        D[tid * 4 + i] = c[i];
}


// =============================================================================
// Example 3: mfma_f32_32x32x16_fp8_fp8 — FP8 input, FP32 output (CDNA3+)
// =============================================================================
// Each of 64 threads holds:
//   A: 8 FP8 elements packed into 64 bits (long)
//   B: 8 FP8 elements packed into 64 bits (long)
//   C/D: 16 FP32 elements

__global__ void mfma_f32_32x32x16_fp8_kernel(const __hip_fp8_storage_t* A,
                                               const __hip_fp8_storage_t* B,
                                               float* C, float* D) {
    int tid = threadIdx.x;

    // Load A: 8 FP8 elements packed into a long (64 bits)
    long a_packed = *reinterpret_cast<const long*>(&A[tid * 8]);

    // Load B: 8 FP8 elements packed into a long
    long b_packed = *reinterpret_cast<const long*>(&B[tid * 8]);

    // Load C accumulator: 16 FP32 elements
    typedef float float16_mfma __attribute__((ext_vector_type(16)));
    float16_mfma c;
    for (int i = 0; i < 16; i++)
        c[i] = C[tid * 16 + i];

    // Execute MFMA: 32x32x16 tile with FP8 inputs
    c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_packed, b_packed, c, 0, 0, 0);

    // Store results
    for (int i = 0; i < 16; i++)
        D[tid * 16 + i] = c[i];
}


// =============================================================================
// Example 4: Block-Scaled MXFP8 — 32x32x64 (CDNA4 / gfx950 only)
// =============================================================================
// Uses mfma_scale intrinsic with E8M0 scale factors per 32-element block.
// Compile with: hipcc --offload-arch=gfx950

#if defined(__gfx950__)
#include <hip/amd_detail/hip_ext_ocp.h>

__global__ void mfma_scale_f32_32x32x64_f8f8_kernel(
    const __amd_fp8_storage_t* A,
    const __amd_fp8_storage_t* B,
    const uint8_t* scale_A,   // E8M0 scale factors for A (32x2 matrix)
    const uint8_t* scale_B,   // E8M0 scale factors for B (2x32 matrix)
    float* C, float* D
) {
    int tid = threadIdx.x;

    // Load A data (64 FP8 elements per thread for 32x32x64)
    // Load B data
    // Load scale factors: each thread holds 1 scale for A and 1 for B
    // actual_scale = 2^(scale_value - 127)

    // [Data loading omitted for brevity — see document.md for full layout]

    typedef float float16_mfma __attribute__((ext_vector_type(16)));
    float16_mfma c;
    for (int i = 0; i < 16; i++)
        c[i] = C[tid * 16 + i];

    // Execute block-scaled MFMA
    // Atype: 0=FP8(E4M3), 1=BF8(E5M2), 2=FP6(E3M2), 3=FP6(E2M3), 4=FP4(E2M1)
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_packed, b_packed, c,
        /*Atype=*/0,     // FP8 (E4M3)
        /*Btype=*/0,     // FP8 (E4M3)
        /*OPSEL_A=*/0,   // First 64-element block of A
        /*OPSEL_B=*/0,   // First 64-element block of B
        scale_a_reg,     // E8M0 scale for A
        scale_b_reg      // E8M0 scale for B
    );

    for (int i = 0; i < 16; i++)
        D[tid * 16 + i] = c[i];
}
#endif


// =============================================================================
// Example 5: Block-Scaled MXFP4 — 32x32x64 (CDNA4 / gfx950 only)
// =============================================================================
// FP4 elements are packed 2 per byte using __amd_fp4x2_storage_t

#if defined(__gfx950__)
__global__ void mfma_scale_f32_32x32x64_f4f4_kernel(
    const __amd_fp4x2_storage_t* A,  // FP4 packed 2 per byte
    const __amd_fp4x2_storage_t* B,
    const uint8_t* scale_A,
    const uint8_t* scale_B,
    float* C, float* D
) {
    int tid = threadIdx.x;

    // FP4 helper functions:
    // __amd_extract_fp4(packed, index) — extract one FP4 from packed byte
    // __amd_create_fp4x2(val0, val1)  — pack two floats into FP4x2

    // Example: extract and inspect FP4 values
    __amd_fp4x2_storage_t packed_val = A[tid];
    float fp4_lo = __amd_extract_fp4(packed_val, 0);  // lower nibble
    float fp4_hi = __amd_extract_fp4(packed_val, 1);  // upper nibble

    typedef float float16_mfma __attribute__((ext_vector_type(16)));
    float16_mfma c;
    for (int i = 0; i < 16; i++)
        c[i] = C[tid * 16 + i];

    // Block-scaled FP4 MFMA
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_packed, b_packed, c,
        /*Atype=*/4,     // FP4 (E2M1)
        /*Btype=*/4,     // FP4 (E2M1)
        /*OPSEL_A=*/0,
        /*OPSEL_B=*/0,
        scale_a_reg,
        scale_b_reg
    );

    for (int i = 0; i < 16; i++)
        D[tid * 16 + i] = c[i];
}
#endif


// =============================================================================
// Host driver (for examples 1-3, compilable on CDNA3)
// =============================================================================
int main() {
    printf("AMD MFMA Matrix Core Examples\n");
    printf("Compile with: hipcc --offload-arch=gfx942 mfma_examples.hip\n");
    printf("For CDNA4 examples, use: hipcc --offload-arch=gfx950\n");

    // Example 1: FP32 32x32x2
    {
        const int M = 32, N = 32, K = 2;
        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_B(K * N, 1.0f);
        std::vector<float> h_C(M * N, 0.0f);
        std::vector<float> h_D(M * N, 0.0f);

        float *d_A, *d_B, *d_C, *d_D;
        hipMalloc(&d_A, M * K * sizeof(float));
        hipMalloc(&d_B, K * N * sizeof(float));
        hipMalloc(&d_C, M * N * sizeof(float));
        hipMalloc(&d_D, M * N * sizeof(float));

        hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_C, h_C.data(), M * N * sizeof(float), hipMemcpyHostToDevice);

        // Launch with single wavefront (64 threads)
        mfma_f32_32x32x2f32_kernel<<<1, 64>>>(d_A, d_B, d_C, d_D);
        hipDeviceSynchronize();

        hipMemcpy(h_D.data(), d_D, M * N * sizeof(float), hipMemcpyDeviceToHost);
        printf("Example 1 (FP32 32x32x2): D[0] = %f (expected 2.0)\n", h_D[0]);

        hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
    }

    return 0;
}
