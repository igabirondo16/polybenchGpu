/**
 * adi.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

//#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU // Changed to CPU for testing

// Ensure comparison against reference CPU version is active for testing
#define RUN_ON_CPU

#include "adi.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0 // Not used in current OpenCL selection logic, but keep for consistency if other files use it

#define MAX_SOURCE_SIZE (0x10000000)

// Alpha parameter for workload splitting. Interpreted as CPU's share.
// 0.0 = GPU (OpenCL dev) takes all, 1.0 = CPU takes all.
double alpha = 0.5;

// OpenCL error checking macro
#define CL_CHECK(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d in file %s line %d\n", err, __FILE__, __LINE__); \
        exit(1); \
    }

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_kernel clKernel4;
cl_kernel clKernel5;
cl_kernel clKernel6;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;
unsigned int mem_size_A;
unsigned int mem_size_B;
unsigned int mem_size_C;

// CPU-side kernel implementations
void adi_cpu_kernel1(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int cpu_rows_start, int cpu_rows_end);
void adi_cpu_kernel2(DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int cpu_rows_start, int cpu_rows_end);
void adi_cpu_kernel3(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int cpu_rows_start, int cpu_rows_end);
void adi_cpu_kernel4(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int i1, int cpu_cols_start, int cpu_cols_end);
void adi_cpu_kernel5(DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int cpu_cols_start, int cpu_cols_end);
void adi_cpu_kernel6(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), int i1, int cpu_cols_start, int cpu_cols_end);


void init_array(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      {
	    X[i][j] = ((DATA_TYPE) i*(j+1) + 1) / N;
	    A[i][j] = ((DATA_TYPE) (i-1)*(j+4) + 2) / N;
	    B[i][j] = ((DATA_TYPE) (i+3)*(j+7) + 3) / N;
      }
}

void compareResults(DATA_TYPE POLYBENCH_2D(B1,N,N,n,n), DATA_TYPE POLYBENCH_2D(B2,N,N,n,n), DATA_TYPE POLYBENCH_2D(X1,N,N,n,n), DATA_TYPE POLYBENCH_2D(X2,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
			if (percentDiff(B1[i][j], B2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(X1[i][j], X2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void read_cl_file()
{
	fp = fopen("adi.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel: adi.cl\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if (!source_str) { fprintf(stderr, "Failed to allocate memory for kernel source.\n"); exit(1); }
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
    if (source_size == 0) {
        fprintf(stderr, "Error: adi.cl is empty or could not be read properly.\n");
        exit(1);
    }
    if (source_size < MAX_SOURCE_SIZE) {
        source_str[source_size] = '\0';
    } else {
        source_str[MAX_SOURCE_SIZE - 1] = '\0';
    }
    // printf("read_cl_file: Read %zu bytes from adi.cl.\n", source_size);
}

void cl_initialization()
{
    // printf("Initializing OpenCL...\n");
    errcode = clGetPlatformIDs(0, NULL, &num_platforms);
    CL_CHECK(errcode);
    if (num_platforms == 0) {
        printf("Error: Failed to find any OpenCL platforms.\n");
        exit(1);
    }
    // printf("Number of available platforms: %d\n", num_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) { fprintf(stderr, "Failed to allocate memory for platforms.\n"); exit(1); }
    errcode = clGetPlatformIDs(num_platforms, platforms, NULL);
    CL_CHECK(errcode);

    platform_id = NULL;
    device_id = NULL;

    for (cl_uint i = 0; i < num_platforms; i++) {
        // errcode = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(str_temp), str_temp, NULL);
        // CL_CHECK(errcode);
        // printf("Platform %d: %s\n", i, str_temp);

        cl_uint num_devices_on_platform;
        errcode = clGetDeviceIDs(platforms[i], OPENCL_DEVICE_SELECTION, 0, NULL, &num_devices_on_platform);
        if (errcode == CL_SUCCESS && num_devices_on_platform > 0) {
            cl_device_id* devices_on_platform = (cl_device_id*)malloc(num_devices_on_platform * sizeof(cl_device_id));
            if(!devices_on_platform) { fprintf(stderr, "Failed to allocate memory for devices_on_platform.\n"); exit(1); }
            errcode = clGetDeviceIDs(platforms[i], OPENCL_DEVICE_SELECTION, num_devices_on_platform, devices_on_platform, NULL);
            if (errcode == CL_SUCCESS) {
                 platform_id = platforms[i];
                 device_id = devices_on_platform[0];
                 num_devices = 1;

                 // errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
                 // CL_CHECK(errcode); printf("Selected platform name is %s\n",str_temp);
                 // errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
                 // CL_CHECK(errcode); printf("Selected device name is %s\n",str_temp);

                 free(devices_on_platform);
                 break;
            }
            free(devices_on_platform);
        }
    }
    free(platforms);

    if (device_id == NULL) {
        printf("Error: No suitable OpenCL device found for type %lu.\n", (unsigned long)OPENCL_DEVICE_SELECTION);
        exit(1);
    }
	
	cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };
	clGPUContext = clCreateContext( props, 1, &device_id, NULL, NULL, &errcode);
	CL_CHECK(errcode);
 
    #if CL_TARGET_OPENCL_VERSION >= 200
        clCommandQue = clCreateCommandQueueWithProperties(clGPUContext, device_id, 0, &errcode);
    #else
        clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
    #endif
	CL_CHECK(errcode);
    // printf("OpenCL Initialized Successfully.\n");
}

void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
	// printf("cl_mem_init: Starting memory initialization...\n");
	mem_size_A = N*N*sizeof(DATA_TYPE);
	mem_size_B = N*N*sizeof(DATA_TYPE);
	mem_size_C = N*N*sizeof(DATA_TYPE);

	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_A, NULL, &errcode); CL_CHECK(errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_B, NULL, &errcode); CL_CHECK(errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_C, NULL, &errcode); CL_CHECK(errcode);
		
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, mem_size_A, A, 0, NULL, NULL); CL_CHECK(errcode);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, mem_size_B, B, 0, NULL, NULL); CL_CHECK(errcode);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, mem_size_C, X, 0, NULL, NULL); CL_CHECK(errcode);
	// printf("cl_mem_init: Memory initialization finished.\n");
}

void cl_load_prog()
{
	// printf("cl_load_prog: Starting program load...\n");
	if (!clGPUContext) { fprintf(stderr, "cl_load_prog: clGPUContext is NULL.\n"); exit(1); }
	if (!source_str) { fprintf(stderr, "cl_load_prog: source_str is NULL.\n"); exit(1); }

	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
	CL_CHECK(errcode);

	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
    if (errcode != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *)malloc(log_size + 1);
        if (!build_log) { perror("Failed to allocate build_log"); exit(1); }
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        fprintf(stderr, "Error in building program: %s\n", build_log);
        free(build_log);
        CL_CHECK(errcode);
    }
		
	clKernel1 = clCreateKernel(clProgram, "adi_kernel1", &errcode); CL_CHECK(errcode);
	clKernel2 = clCreateKernel(clProgram, "adi_kernel2", &errcode); CL_CHECK(errcode);
	clKernel3 = clCreateKernel(clProgram, "adi_kernel3", &errcode); CL_CHECK(errcode);
	clKernel4 = clCreateKernel(clProgram, "adi_kernel4", &errcode); CL_CHECK(errcode);
	clKernel5 = clCreateKernel(clProgram, "adi_kernel5", &errcode); CL_CHECK(errcode);
	clKernel6 = clCreateKernel(clProgram, "adi_kernel6", &errcode); CL_CHECK(errcode);

    errcode = clFinish(clCommandQue); CL_CHECK(errcode);
	// printf("cl_load_prog: Program load finished.\n");
}

// Common launcher for kernels 1,2,3 (row-wise) and 4,5,6 (col-wise for a given row i1)
void cl_launch_kernel_common(cl_kernel kernel, int num_work_items, int i1_val_for_k4_k6)
{
    if (num_work_items <= 0) return;

    size_t local_ws[1] = {DIM_LOCAL_WORK_GROUP_X};
    size_t global_ws[1] = {(size_t)ceil(((float)num_work_items) / ((float)local_ws[0])) * local_ws[0]};

    // printf("Attempting to launch kernel %p: work_items=%d, global_ws[0]=%zu, local_ws[0]=%zu, i1_val=%d\n", kernel, num_work_items, global_ws[0], local_ws[0], i1_val_for_k4_k6);

    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj); CL_CHECK(errcode);
    errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj); CL_CHECK(errcode);
    errcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj); CL_CHECK(errcode);

    if (kernel == clKernel4 || kernel == clKernel6) {
        errcode = clSetKernelArg(kernel, 3, sizeof(int), (void *)&i1_val_for_k4_k6); CL_CHECK(errcode);
    }
    // Kernels in adi.cl define N via preprocessor, so no N argument needed here.

    errcode = clEnqueueNDRangeKernel(clCommandQue, kernel, 1, NULL, global_ws, local_ws, 0, NULL, NULL);
    CL_CHECK(errcode);
    // printf("Kernel enqueued.\n");
}

void cl_launch_kernel1(int num_gpu_rows) { cl_launch_kernel_common(clKernel1, num_gpu_rows, 0); }
void cl_launch_kernel2(int num_gpu_rows) { cl_launch_kernel_common(clKernel2, num_gpu_rows, 0); }
void cl_launch_kernel3(int num_gpu_rows) { cl_launch_kernel_common(clKernel3, num_gpu_rows, 0); }
void cl_launch_kernel4(int i1, int num_gpu_cols) { cl_launch_kernel_common(clKernel4, num_gpu_cols, i1); }
void cl_launch_kernel5(int num_gpu_cols) { cl_launch_kernel_common(clKernel5, num_gpu_cols, 0); } // i1 not used by kernel5
void cl_launch_kernel6(int i1_loop_var, int num_gpu_cols) { cl_launch_kernel_common(clKernel6, num_gpu_cols, i1_loop_var); }


void cl_clean_up()
{
	// printf("cl_clean_up: Starting cleanup...\n");
	errcode = clFlush(clCommandQue); CL_CHECK(errcode);
	errcode = clFinish(clCommandQue); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel1); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel2); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel3); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel4); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel5); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel6); CL_CHECK(errcode);
	errcode = clReleaseProgram(clProgram); CL_CHECK(errcode);
	errcode = clReleaseMemObject(a_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseMemObject(b_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseMemObject(c_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseCommandQueue(clCommandQue); CL_CHECK(errcode);
	errcode = clReleaseContext(clGPUContext); CL_CHECK(errcode);
	// printf("cl_clean_up: Cleanup finished.\n");
}

// Reference CPU implementation for comparison
void adi_ref(DATA_TYPE POLYBENCH_2D(A_ref,N,N,n_ref,n_ref2), DATA_TYPE POLYBENCH_2D(B_ref,N,N,n_b_ref,n_b_ref2), DATA_TYPE POLYBENCH_2D(X_ref,N,N,n_x_ref,n_x_ref2))
{
	int t;
	int i1;
	int i2;

	for (t = 0; t < TSTEPS; t++)
    {
        for (i1 = 0; i1 < N; i1++)
        {
            for (i2 = 1; i2 < N; i2++)
            {
                X_ref[i1][i2] = X_ref[i1][i2] - X_ref[i1][(i2-1)] * A_ref[i1][i2] / B_ref[i1][(i2-1)];
                B_ref[i1][i2] = B_ref[i1][i2] - A_ref[i1][i2] * A_ref[i1][i2] / B_ref[i1][(i2-1)];
            }
        }

        for (i1 = 0; i1 < N; i1++)
        {
            X_ref[i1][(N-1)] = X_ref[i1][(N-1)] / B_ref[i1][(N-1)];
        }

        for (i1 = 0; i1 < N; i1++)
        {
            for (i2 = 0; i2 < N-2; i2++)
            {
                 X_ref[i1][(N-i2-2)] = (X_ref[i1][(N-2-i2)] - X_ref[i1][(N-2-i2-1)] * A_ref[i1][(N-i2-3)]) / B_ref[i1][(N-3-i2)];
            }
        }
        for (i1 = 1; i1 < N; i1++)
        {
            for (i2 = 0; i2 < N; i2++)
            {
                X_ref[i1][i2] = X_ref[i1][i2] - X_ref[(i1-1)][i2] * A_ref[i1][i2] / B_ref[(i1-1)][i2];
                B_ref[i1][i2] = B_ref[i1][i2] - A_ref[i1][i2] * A_ref[i1][i2] / B_ref[(i1-1)][i2];
            }
        }

        for (i2 = 0; i2 < N; i2++)
        {
            X_ref[(N-1)][i2] = X_ref[(N-1)][i2] / B_ref[(N-1)][i2];
        }

        for (i1 = 0; i1 < N-2; i1++)
        {
            for (i2 = 0; i2 < N; i2++)
            {
                 X_ref[(N-2-i1)][i2] = (X_ref[(N-2-i1)][i2] - X_ref[(N-i1-3)][i2] * A_ref[(N-3-i1)][i2]) / B_ref[(N-2-i1)][i2];
            }
        }
    }
}

// CPU-side kernel implementations for collaborative version
void adi_cpu_kernel1(DATA_TYPE POLYBENCH_2D(A,N,N,n_dim,n_dim2), DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int cpu_rows_start, int cpu_rows_end)
{
    int i1, i2;
    for (i1 = cpu_rows_start; i1 < cpu_rows_end; i1++)
    {
        for (i2 = 1; i2 < N; i2++)
        {
            X[i1][i2] = X[i1][i2] - X[i1][(i2-1)] * A[i1][i2] / B[i1][(i2-1)];
            B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][(i2-1)];
        }
    }
}

void adi_cpu_kernel2(DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int cpu_rows_start, int cpu_rows_end)
{
    int i1;
    for (i1 = cpu_rows_start; i1 < cpu_rows_end; i1++)
    {
        X[i1][(N-1)] = X[i1][(N-1)] / B[i1][(N-1)];
    }
}

void adi_cpu_kernel3(DATA_TYPE POLYBENCH_2D(A,N,N,n_dim,n_dim2), DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int cpu_rows_start, int cpu_rows_end)
{
    int i1, i2;
    for (i1 = cpu_rows_start; i1 < cpu_rows_end; i1++)
    {
        for (i2 = 0; i2 < N-2; i2++)
        {
             X[i1][N-i2-2] = (X[i1][N-2-i2] - X[i1][N-2-i2-1] * A[i1][N-i2-3]) / B[i1][N-3-i2];
        }
    }
}

void adi_cpu_kernel4(DATA_TYPE POLYBENCH_2D(A,N,N,n_dim,n_dim2), DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int i1, int cpu_cols_start, int cpu_cols_end)
{
    int i2;
    for (i2 = cpu_cols_start; i2 < cpu_cols_end; i2++)
    {
        X[i1][i2] = X[i1][i2] - X[(i1-1)][i2] * A[i1][i2] / B[(i1-1)][i2];
        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[(i1-1)][i2];
    }
}

void adi_cpu_kernel5(DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int cpu_cols_start, int cpu_cols_end)
{
    int i2;
    for (i2 = cpu_cols_start; i2 < cpu_cols_end; i2++)
    {
        X[(N-1)][i2] = X[(N-1)][i2] / B[(N-1)][i2];
    }
}

void adi_cpu_kernel6(DATA_TYPE POLYBENCH_2D(A,N,N,n_dim,n_dim2), DATA_TYPE POLYBENCH_2D(B,N,N,n_dim_b,n_dim2_b), DATA_TYPE POLYBENCH_2D(X,N,N,n_dim_x,n_dim2_x), int i1_loop_var, int cpu_cols_start, int cpu_cols_end)
{
    int i2;
    int actual_row = N-2-i1_loop_var;
    for (i2 = cpu_cols_start; i2 < cpu_cols_end; i2++)
    {
        X[actual_row][i2] = (X[actual_row][i2] - X[N-i1_loop_var-3][i2] * A[N-3-i1_loop_var][i2]) / B[actual_row][i2];
    }
}

static
void print_array(int n_print, DATA_TYPE POLYBENCH_2D(X_pa,N,N,nx_pa,ny_pa)) // Use X_pa to avoid conflict with global X if any
{
  int i, j;
  for (i = 0; i < n_print; i++)
    for (j = 0; j < n_print; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X_pa[i][j]);
      if ((i * N + j) % 20 == 0) fprintf(stderr, "\n"); // N from header
    }
  fprintf(stderr, "\n");
}

int main(int argc, char *argv[])
{
	if (argc > 1) {
		alpha = atof(argv[1]);
		if (alpha < 0.0 || alpha > 1.0) {
			fprintf(stderr, "ERROR: Alpha value (CPU share) must be between 0.0 and 1.0. Defaulting to 0.5.\n");
			alpha = 0.5;
		}
	}
	printf("Current ADI alpha (CPU Share) = %f\n", alpha);

	POLYBENCH_2D_ARRAY_DECL(A1,DATA_TYPE,N,N,n,n); // For combined CPU/GPU work
	POLYBENCH_2D_ARRAY_DECL(A2,DATA_TYPE,N,N,n,n); // For reference CPU-only work
	POLYBENCH_2D_ARRAY_DECL(B1,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B2,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X1,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X2,DATA_TYPE,N,N,n,n);

	init_array(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1));
	init_array(POLYBENCH_ARRAY(A2), POLYBENCH_ARRAY(B2), POLYBENCH_ARRAY(X2));

	// Only initialize OpenCL if GPU is doing some work
	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1));
	cl_load_prog();
	
	int total_rows_k123 = N;
    int cpu_start = (int)((1.0f - alpha) * total_rows_k123);
	int cpu_end = total_rows_k123;

	int cpu_rows_k123 = cpu_end - cpu_start;
	int gpu_rows_k123 = total_rows_k123 - cpu_rows_k123;

	int total_cols_k456 = N;
	int cpu_cols_k456 = cpu_end - cpu_start;
	int gpu_cols_k456 = total_cols_k456 - cpu_cols_k456;
	
	 printf("TSTEPS = %d\n", TSTEPS);
     printf("N = %d\n", N);
	 printf("Alpha = %f, K123: GPU rows = %d, CPU rows = %d (start %d)\n", alpha, gpu_rows_k123, cpu_rows_k123, gpu_rows_k123);
	 printf("Alpha = %f, K456: GPU cols = %d, CPU cols = %d (start %d)\n", alpha, gpu_cols_k456, cpu_cols_k456, gpu_cols_k456);

  	polybench_start_instruments;

	int t, i1;
	for (t = 0; t < TSTEPS; t++)
	{
		// Kernel 1
		if (gpu_rows_k123 > 0) {
			cl_launch_kernel1(gpu_rows_k123);
		}
		if (cpu_rows_k123 > 0) {
			adi_cpu_kernel1(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), cpu_start, cpu_end);
			
        }    

        errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        if (gpu_rows_k123 > 0 && cpu_rows_k123 > 0) {
            // Write contents from GPU to CPU
            errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, gpu_rows_k123 * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(X1), 0, NULL, NULL); CL_CHECK(errcode);
            errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, gpu_rows_k123 * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(B1), 0, NULL, NULL); CL_CHECK(errcode);

            size_t offset = cpu_start * N * sizeof(DATA_TYPE);

            DATA_TYPE *c_ptr = (DATA_TYPE*) POLYBENCH_ARRAY(X1) + (cpu_start * N);
            DATA_TYPE *b_ptr = (DATA_TYPE*) POLYBENCH_ARRAY(B1) + (cpu_start * N);

            // Write contents from CPU to GPU
            errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, offset, cpu_rows_k123 * N * sizeof(DATA_TYPE), c_ptr, 0, NULL, NULL); CL_CHECK(errcode);
            errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, offset, cpu_rows_k123 * N * sizeof(DATA_TYPE), b_ptr, 0, NULL, NULL); CL_CHECK(errcode);
        
            errcode = clFinish(clCommandQue); CL_CHECK(errcode);
        }

                
		// Kernel 2
		if (gpu_rows_k123 > 0) {
            cl_launch_kernel2(gpu_rows_k123);
            
        } 
		if (cpu_rows_k123 > 0) {
			adi_cpu_kernel2(POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), cpu_start, cpu_end);
		
		}
        errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        if (gpu_rows_k123 > 0 && cpu_rows_k123 > 0) {
            // Write contents from GPU to CPU
            errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, gpu_rows_k123 * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(X1), 0, NULL, NULL); CL_CHECK(errcode);

            size_t offset = cpu_start * N * sizeof(DATA_TYPE);
            DATA_TYPE *c_ptr = (DATA_TYPE*) POLYBENCH_ARRAY(X1) + (cpu_start * N);

            // Write contents from CPU to GPU
            errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, offset, cpu_rows_k123 * N * sizeof(DATA_TYPE), c_ptr, 0, NULL, NULL); CL_CHECK(errcode);
            errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        }
		// Kernel 3
		if (gpu_rows_k123 > 0) {
            cl_launch_kernel3(gpu_rows_k123);
        
        }
		if (cpu_rows_k123 > 0) {
			adi_cpu_kernel3(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), cpu_start, cpu_end);
			
		}
        errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        if (gpu_rows_k123 > 0 && cpu_rows_k123 > 0) {
            // Write contents from GPU to CPU
            errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, gpu_rows_k123 * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(X1), 0, NULL, NULL); CL_CHECK(errcode);

            size_t offset = cpu_start * N * sizeof(DATA_TYPE);
            DATA_TYPE *c_ptr = (DATA_TYPE*) POLYBENCH_ARRAY(X1) + (cpu_start * N);

            // Write contents from CPU to GPU
            errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, offset, cpu_rows_k123 * N * sizeof(DATA_TYPE), c_ptr, 0, NULL, NULL); CL_CHECK(errcode);
            errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        }

		// Kernel 4
		for (i1 = 1; i1 < N; i1++)
		{
			if (gpu_cols_k456 > 0) {
                cl_launch_kernel4(i1, gpu_cols_k456);

            }
			if (cpu_cols_k456 > 0) {
				adi_cpu_kernel4(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), i1, gpu_cols_k456, total_cols_k456);
			
            }

			errcode = clFinish(clCommandQue); CL_CHECK(errcode); 

            if (gpu_cols_k456 > 0 && cpu_cols_k456) {
                size_t row_offset = i1*N * sizeof(DATA_TYPE);
                size_t col_offset = row_offset + (cpu_start * sizeof(DATA_TYPE));
                size_t cols_read_bytes = gpu_cols_k456 * sizeof(DATA_TYPE);

                DATA_TYPE * c_col1 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (i1*N);
                DATA_TYPE * b_col1 = (DATA_TYPE*)POLYBENCH_ARRAY(B1) + (i1*N); 

                // Read from GPU to CPU
                errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, row_offset, cols_read_bytes, c_col1, 0, NULL, NULL); CL_CHECK(errcode);
                errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, row_offset, cols_read_bytes, b_col1, 0, NULL, NULL); CL_CHECK(errcode);

                // Write from CPU to GPU
                DATA_TYPE * c_col2 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (i1*N) + cpu_start;
                DATA_TYPE * b_col2 = (DATA_TYPE*)POLYBENCH_ARRAY(B1) + (i1*N) + cpu_start; 

                // Write contents from CPU to GPU
                errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, col_offset, cpu_cols_k456 * sizeof(DATA_TYPE), c_col2, 0, NULL, NULL); CL_CHECK(errcode);
                errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, col_offset, cpu_cols_k456 * sizeof(DATA_TYPE), b_col2, 0, NULL, NULL); CL_CHECK(errcode);
                errcode = clFinish(clCommandQue); CL_CHECK(errcode);

            }

		}

        
		// Kernel 5
        int i1_k5 = N-1;
        if (gpu_cols_k456 > 0) {
            cl_launch_kernel5(gpu_cols_k456);
        
        }
        if (cpu_cols_k456 > 0) {
            adi_cpu_kernel5(POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), gpu_cols_k456, total_cols_k456);
        
        }

        errcode = clFinish(clCommandQue); CL_CHECK(errcode);

        if (gpu_cols_k456 > 0 && cpu_cols_k456 > 0) {
            size_t row_offset = i1_k5*N * sizeof(DATA_TYPE);
            size_t col_offset = row_offset + (cpu_start * sizeof(DATA_TYPE));
            size_t cols_read_bytes = gpu_cols_k456 * sizeof(DATA_TYPE);

            DATA_TYPE * c_col1 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (i1_k5*N);

            // Read from GPU to CPU
            errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, row_offset, cols_read_bytes, c_col1, 0, NULL, NULL); CL_CHECK(errcode);

            // Write from CPU to GPU
            DATA_TYPE * c_col2 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (i1_k5*N) + cpu_start;

            // Write contents from CPU to GPU
            errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, col_offset, cpu_cols_k456 * sizeof(DATA_TYPE), c_col2, 0, NULL, NULL); CL_CHECK(errcode);
            errcode = clFinish(clCommandQue); CL_CHECK(errcode);
        } 
		
        
		// Kernel 6
		for (int i1_loop = 0; i1_loop < N-2; i1_loop++)
		{
			int actual_row = N-2-i1_loop;
			if (gpu_cols_k456 > 0) {
                cl_launch_kernel6(i1_loop, gpu_cols_k456);
            
            }
			if (cpu_cols_k456 > 0) {
				adi_cpu_kernel6(POLYBENCH_ARRAY(A1), POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(X1), i1_loop, gpu_cols_k456, total_cols_k456);
			}
			errcode = clFinish(clCommandQue); CL_CHECK(errcode); 

            if (gpu_cols_k456 > 0 && cpu_cols_k456 > 0) {
                size_t row_offset = actual_row*N * sizeof(DATA_TYPE);
                size_t col_offset = row_offset + (cpu_start * sizeof(DATA_TYPE));
                size_t cols_read_bytes = gpu_cols_k456 * sizeof(DATA_TYPE);

                DATA_TYPE * c_col1 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (actual_row*N);

                // Read from GPU to CPU
                errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, row_offset, cols_read_bytes, c_col1, 0, NULL, NULL); CL_CHECK(errcode);

                // Write from CPU to GPU
                DATA_TYPE * c_col2 = (DATA_TYPE*)POLYBENCH_ARRAY(X1) + (actual_row*N) + cpu_start;

                // Write contents from CPU to GPU
                errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, col_offset, cpu_cols_k456 * sizeof(DATA_TYPE), c_col2, 0, NULL, NULL); CL_CHECK(errcode);
                errcode = clFinish(clCommandQue); CL_CHECK(errcode);
            } 
		}
        
	}	
	
    if (alpha < 1.0) {
        // printf("Final host sync: Reading full X1 and B1 from device.\n");
        errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, mem_size_C, POLYBENCH_ARRAY(X1), 0, NULL, NULL); CL_CHECK(errcode);
        errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, mem_size_B, POLYBENCH_ARRAY(B1), 0, NULL, NULL); CL_CHECK(errcode);
    }
	if (alpha < 1.0) { // Only finish if OpenCL was used
        clFinish(clCommandQue); CL_CHECK(errcode);
    }

    printf("\nCPU-GPU Time in seconds: ");
	polybench_stop_instruments;
	polybench_print_instruments;

    size_t total_bytes = mem_size_A + mem_size_B + mem_size_C;
	printf("Total bytes: %ld\n", total_bytes);

	size_t wg_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
	printf("Work group size: %ld\n", wg_size);
	
	#ifdef RUN_ON_CPU
		/* Start timer. */
	  	polybench_start_instruments;
		adi_ref(POLYBENCH_ARRAY(A2), POLYBENCH_ARRAY(B2), POLYBENCH_ARRAY(X2));
		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
		compareResults(POLYBENCH_ARRAY(B1), POLYBENCH_ARRAY(B2), POLYBENCH_ARRAY(X1), POLYBENCH_ARRAY(X2));
	#else
		if (polybench_get_dump_array_setting()) {
            printf("Printing A1, B1, X1 to stderr\n");
            print_array(N, POLYBENCH_ARRAY(A1)); // N from header for print_array
            print_array(N, POLYBENCH_ARRAY(B1));
		    print_array(N, POLYBENCH_ARRAY(X1));
        }
	#endif

	if (alpha < 1.0) {
	    cl_clean_up();
    }

	POLYBENCH_FREE_ARRAY(A1);
	POLYBENCH_FREE_ARRAY(A2);
	POLYBENCH_FREE_ARRAY(B1);
	POLYBENCH_FREE_ARRAY(B2);
	POLYBENCH_FREE_ARRAY(X1);
	POLYBENCH_FREE_ARRAY(X2);

    return 0;
}

#include "../../common/polybench.c"
