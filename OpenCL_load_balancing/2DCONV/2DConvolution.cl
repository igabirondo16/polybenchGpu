/**
 * 2DConvolution.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

__kernel void Convolution2D_kernel(__global DATA_TYPE *A, __global DATA_TYPE *B, int ni, int nj, int start_row)
{
	// Calculate global position
	int i = get_global_id(0) + start_row; // Row index (offset by start_row)
	int j = get_global_id(1);             // Column index

	// Filter coefficients
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
	
	// Only perform convolution for valid positions (excluding borders)
	if (i > 0 && j > 0 && i < ni - 1 && j < nj - 1)
	{
		// Convert 2D indices to 1D memory index: i*nj + j
		int idx = i * nj + j;
		
		// Calculate convolution with boundary checks
		B[idx] = c11 * A[(i - 1) * nj + (j - 1)] + 
		         c21 * A[(i - 1) * nj + j] + 
		         c31 * A[(i - 1) * nj + (j + 1)] +
		         c12 * A[i * nj + (j - 1)] + 
		         c22 * A[i * nj + j] + 
		         c32 * A[i * nj + (j + 1)] +
		         c13 * A[(i + 1) * nj + (j - 1)] + 
		         c23 * A[(i + 1) * nj + j] + 
		         c33 * A[(i + 1) * nj + (j + 1)];
	}
}
