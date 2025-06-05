/**
 * gramschmidt.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

typedef float DATA_TYPE;

// Kernel 1: Compute partial sum for R[k][k] (norm of A[:,k])
// A: input matrix
// k: current column index
// m_rows_for_gpu: number of rows this GPU launch is responsible for
// n_total_cols: total number of columns in A (original N)
// partial_sum_output: buffer to store partial sum computed by this work-group / kernel instance
// For simplicity, this kernel assumes it's launched with enough work-items,
// and reduction to a single value per call will be handled by work-group logic or host.
// A common pattern: each work-item computes a value, then reduce in local memory, one work-item writes result.
// Here, let's make it simple: one work-group computes one partial sum.
// The host will then sum these few partial sums if num_work_groups > 1.
// Or, if global size is small enough (e.g. 256), a single workgroup reduces.
__kernel void kernel1_norm_partial_sum_gpu(__global const DATA_TYPE *A,
                                           int k,
                                           int m_rows_for_gpu,
                                           int n_total_cols,
                                           __global DATA_TYPE *partial_sum_output,
                                           __local DATA_TYPE *local_sums)
{
	int local_id = get_local_id(0);
	int group_id = get_group_id(0);
	int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    DATA_TYPE acc = 0.0;
    // Each work-item computes sum for a subset of its rows
    for (int i = global_id; i < m_rows_for_gpu; i += get_global_size(0)) {
        DATA_TYPE val = A[i * n_total_cols + k];
        acc += val * val;
    }
    local_sums[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sums[local_id] += local_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in group writes to global partial sum output
    if (local_id == 0) {
        partial_sum_output[group_id] = local_sums[0];
    }
}

// Kernel 2: Compute Q[:,k] = A[:,k] / R[k][k]
// A: input matrix (current state)
// Q: output matrix
// Rkk_value: scalar R[k][k] computed on host
// k: current column index
// m_rows_for_gpu: number of rows this GPU launch is responsible for
// n_total_cols: total number of columns (original N)
__kernel void kernel2_compute_q_col_gpu(__global const DATA_TYPE *A,
                                        __global DATA_TYPE *Q,
                                        DATA_TYPE Rkk_value,
                                        int k,
                                        int m_rows_for_gpu,
                                        int n_total_cols)
{
	int i = get_global_id(0); // Row index for this GPU portion

    if (i < m_rows_for_gpu)
	{
		// A is the matrix being modified in place by CPU/GPU for its parts.
        // Q is the matrix Q. For Q[:,k], A[:,k] is the source *before* R[k][k] is used for A itself.
        // This means A passed here must be the version *before* it's modified by Q[:,k]*R[k][j] in step 3b.
        // The host must ensure A_mem_obj has the correct version of A for column k.
		Q[i * n_total_cols + k] = A[i * n_total_cols + k] / Rkk_value;
	}
}

// Kernel 3a: Compute partial sum for R[k][j] = Q[:,k] . A[:,j]
// Q: input matrix Q (specifically column k)
// A: input matrix A (specifically column j_col)
// k: current main column index (for Q)
// j_col: the column index in A for the dot product
// m_rows_for_gpu: number of rows this GPU launch is responsible for
// n_total_cols: total number of columns (original N, same for Q and A)
// partial_sum_output: buffer to store partial sum
__kernel void kernel3_Rkj_partial_sum_gpu(__global const DATA_TYPE *Q,
                                          __global const DATA_TYPE *A,
                                          int k,
                                          int j_col,
                                          int m_rows_for_gpu,
                                          int n_total_cols,
                                          __global DATA_TYPE *partial_sum_output,
                                          __local DATA_TYPE *local_sums)
{
    int local_id = get_local_id(0);
	int group_id = get_group_id(0);
	int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    DATA_TYPE acc = 0.0;
    for (int i = global_id; i < m_rows_for_gpu; i += get_global_size(0)) {
        acc += Q[i * n_total_cols + k] * A[i * n_total_cols + j_col];
    }
    local_sums[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sums[local_id] += local_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        partial_sum_output[group_id] = local_sums[0];
    }
}

// Kernel 3b: Update A[:,j] = A[:,j] - Q[:,k] * R[k][j]
// A: matrix to be updated (column j_col)
// Q: input matrix Q (specifically column k)
// Rkj_value: scalar R[k][j] computed on host
// k: current main column index (for Q)
// j_col: the column index in A to update
// m_rows_for_gpu: number of rows this GPU launch is responsible for
// n_total_cols: total number of columns (original N)
__kernel void kernel3_update_Aij_gpu(__global DATA_TYPE *A,
                                     __global const DATA_TYPE *Q,
                                     DATA_TYPE Rkj_value,
                                     int k,
                                     int j_col,
                                     int m_rows_for_gpu,
                                     int n_total_cols)
{
	int i = get_global_id(0); // Row index for this GPU portion

	if (i < m_rows_for_gpu)
	{
		A[i * n_total_cols + j_col] = A[i * n_total_cols + j_col] - Q[i * n_total_cols + k] * Rkj_value;
	}
}

// Original kernels (can be removed if new ones are confirmed to work)
/*
__kernel void gramschmidt_kernel1(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int tid = get_global_id(0);
	
	if (tid == 0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < m; i++)
		{
			nrm += a[i * n + k] * a[i * n + k];
		}
      		r[k * n + k] = sqrt(nrm);
	}
}


__kernel void gramschmidt_kernel2(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int i = get_global_id(0);

        if (i < m)
	{	
		q[i * n + k] = a[i * n + k] / r[k * n + k];
	}
}


__kernel void gramschmidt_kernel3(__global DATA_TYPE *a, __global DATA_TYPE *r, __global DATA_TYPE *q, int k, int m, int n)
{
	int j = get_global_id(0) + (k+1);

	if ((j < n))
	{
		r[k*n + j] = 0.0;

		int i;
		for (i = 0; i < m; i++)
		{
			r[k*n + j] += q[i*n + k] * a[i*n + j];
		}
		
		for (i = 0; i < m; i++)
		{
			a[i*n + j] -= q[i*n + k] * r[k*n + j];
		}
	}
}
*/
