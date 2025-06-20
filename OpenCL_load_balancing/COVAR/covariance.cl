/**
 * covariance.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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


__kernel void mean_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *data, DATA_TYPE float_n, int m, int n)
{
	int j = get_global_id(0);

	if (j < m)
	{
		mean[j] = 0.0;

		int i;
		for(i = 0; i < n; i++)
		{
			mean[j] += data[i * m + j];
		}
		mean[j] /= (DATA_TYPE)float_n;
	}
}

__kernel void reduce_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *data, int m, int n)
{
	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < n) && (j < m))
	{
		data[i * m + j] -= mean[j];
	}
}


__kernel void covar_kernel(__global DATA_TYPE *symmat, __global DATA_TYPE *data, int m, int n)
{
	int j1 = get_global_id(0);
	int i, j2;

	if (j1 < m) // Iterate over columns j1 (features)
	{
		for (j2 = j1; j2 < m; j2++) // Iterate over columns j2 (features), starting from j1
		{
			DATA_TYPE sum = 0.0f; // Use DATA_TYPE for sum
			for(i = 0; i < n; i++) // Iterate over data points (rows)
			{
				// Access data as data[row_idx * num_cols + col_idx]
				// num_cols is m (M_features)
				// So, data[i * m + j1] and data[i * m + j2]
				sum += data[i * m + j1] * data[i * m + j2];
			}

            // Perform normalization
            if (n > 1)
            {
			    symmat[j1 * m + j2] = sum / (n - 1.0f);
            }
            else
            {
                symmat[j1 * m + j2] = sum; // Or 0.0f, if n=1 covariance is just the sum (product) or undefined.
                                          // C code implies it calculates sum/(0.0) if n=1.
                                          // This path (n=1) is unlikely for typical datasets.
            }

            // Assign to the symmetric element
			//symmat[j2 * m + j1] = symmat[j1 * m + j2];
		}
	}
}
