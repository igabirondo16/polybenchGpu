/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

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
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "gramschmidt.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

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
// cl_kernel clKernel1; // Original, will be replaced by more specific kernels
// cl_kernel clKernel2;
// cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_kernel clKernel1_norm_sum;
cl_kernel clKernel2_q_col;
cl_kernel clKernel3_Rkj_sum;
cl_kernel clKernel3_update_A;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem q_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

//#define RUN_ON_CPU

// Alpha for workload splitting (0.0 to 1.0 for GPU percentage)
double alpha_split = 0.5;


void compareResults(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,M,N,m,n))
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("gramschmidt.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(DATA_TYPE POLYBENCH_2D(A,M,N,m,n))
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(Q,M,N,m_q,n_q), DATA_TYPE POLYBENCH_2D(R,N,N,n_r_rows,n_r_cols))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode); // Q is M*N
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode); // R is N*N
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	// Write initial A data. Q and R are initially computed, so host versions can be zero.
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, A, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, Q, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, R, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing initial buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
	if(errcode != CL_SUCCESS) { printf("Error in creating program\n"); exit(1); }

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log_str = (char *)malloc(log_size); // Renamed log to log_str to avoid conflict
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, log_str, NULL);
        printf("%s\n", log_str);
        free(log_str);
        printf("Error in building program\n");
        exit(1);
    }
		
	clKernel1_norm_sum = clCreateKernel(clProgram, "kernel1_norm_partial_sum_gpu", &errcode);
	if(errcode != CL_SUCCESS) { printf("Error in creating kernel1_norm_partial_sum_gpu\n"); exit(1); }

	clKernel2_q_col = clCreateKernel(clProgram, "kernel2_compute_q_col_gpu", &errcode);
	if(errcode != CL_SUCCESS) { printf("Error in creating kernel2_compute_q_col_gpu\n"); exit(1); }

	clKernel3_Rkj_sum = clCreateKernel(clProgram, "kernel3_Rkj_partial_sum_gpu", &errcode);
	if(errcode != CL_SUCCESS) { printf("Error in creating kernel3_Rkj_partial_sum_gpu\n"); exit(1); }

	clKernel3_update_A = clCreateKernel(clProgram, "kernel3_update_Aij_gpu", &errcode);
	if(errcode != CL_SUCCESS) { printf("Error in creating kernel3_update_Aij_gpu\n"); exit(1); }

	clFinish(clCommandQue);
}

// Define a maximum number of work-groups for partial sum collection
// This should be large enough for typical GPU configurations with chosen work-group sizes.
// E.g., if M is up to 4096 and local_size is 256, max groups = 16.
// If M is larger or local_size smaller, this might need adjustment or dynamic allocation.
#define MAX_PARTIAL_SUM_GROUPS 64


// Launcher for kernel1_norm_partial_sum_gpu
// Returns the sum of partial sums from GPU work-groups.
DATA_TYPE launch_kernel1_norm_partial_sum_gpu(int k, int m_rows_for_gpu, int n_total_cols, int local_size_x)
{
    if (m_rows_for_gpu <= 0) return 0.0;

    size_t localWorkSize[1] = {(size_t)local_size_x};
    // Calculate number of groups needed. Each group processes local_size_x elements "at once" effectively for the reduction.
    // The kernel iterates if m_rows_for_gpu > global_size.
    // Global size should be a multiple of local_size_x, covering m_rows_for_gpu.
    // For reduction, it's often best if one workgroup handles one reduction if data fits, or launch enough workgroups.
    // Let's assume m_rows_for_gpu is reasonably small for one call or we want to sum up workgroup results.
    // Number of work-groups:
    size_t num_groups = (m_rows_for_gpu + local_size_x - 1) / local_size_x;
    if (num_groups > MAX_PARTIAL_SUM_GROUPS) num_groups = MAX_PARTIAL_SUM_GROUPS; // Cap for fixed-size buffer
    if (num_groups == 0 && m_rows_for_gpu > 0) num_groups = 1;


    size_t globalWorkSize[1] = {num_groups * local_size_x};

    // Create buffer for partial sums from GPU (one sum per work-group)
    cl_mem partial_sum_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, num_groups * sizeof(DATA_TYPE), NULL, &errcode);
    if (errcode != CL_SUCCESS) { printf("Error creating buffer for partial sums K1\n"); return 0.0; }

    errcode =  clSetKernelArg(clKernel1_norm_sum, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    errcode |= clSetKernelArg(clKernel1_norm_sum, 1, sizeof(int), (void *)&k);
    errcode |= clSetKernelArg(clKernel1_norm_sum, 2, sizeof(int), (void *)&m_rows_for_gpu);
    errcode |= clSetKernelArg(clKernel1_norm_sum, 3, sizeof(int), (void *)&n_total_cols);
    errcode |= clSetKernelArg(clKernel1_norm_sum, 4, sizeof(cl_mem), (void *)&partial_sum_mem_obj);
    errcode |= clSetKernelArg(clKernel1_norm_sum, 5, local_size_x * sizeof(DATA_TYPE), NULL); // Local memory for reduction
    if(errcode != CL_SUCCESS) { printf("Error setting args for K1_norm_sum: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1_norm_sum, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error launching K1_norm_sum: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    DATA_TYPE partial_sums_host[MAX_PARTIAL_SUM_GROUPS]; // Max size
    errcode = clEnqueueReadBuffer(clCommandQue, partial_sum_mem_obj, CL_TRUE, 0, num_groups * sizeof(DATA_TYPE), partial_sums_host, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error reading partial sums K1: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    clReleaseMemObject(partial_sum_mem_obj);

    DATA_TYPE total_gpu_sum = 0.0;
    for (int i = 0; i < num_groups; i++) {
        total_gpu_sum += partial_sums_host[i];
    }
    return total_gpu_sum;
}

// Launcher for kernel2_compute_q_col_gpu
void launch_kernel2_compute_q_col_gpu(DATA_TYPE Rkk_value, int k, int m_rows_for_gpu, int n_total_cols, int local_size_x)
{
    if (m_rows_for_gpu <= 0) return;

    size_t localWorkSize[1] = {(size_t)local_size_x};
    size_t globalWorkSize[1] = {(size_t)ceil(((float)m_rows_for_gpu) / local_size_x) * local_size_x};

    errcode =  clSetKernelArg(clKernel2_q_col, 0, sizeof(cl_mem), (void *)&a_mem_obj); // A is source for Q[i][k]
    errcode |= clSetKernelArg(clKernel2_q_col, 1, sizeof(cl_mem), (void *)&q_mem_obj); // Q is destination
    errcode |= clSetKernelArg(clKernel2_q_col, 2, sizeof(DATA_TYPE), (void *)&Rkk_value);
    errcode |= clSetKernelArg(clKernel2_q_col, 3, sizeof(int), (void *)&k);
    errcode |= clSetKernelArg(clKernel2_q_col, 4, sizeof(int), (void *)&m_rows_for_gpu);
    errcode |= clSetKernelArg(clKernel2_q_col, 5, sizeof(int), (void *)&n_total_cols);
    if(errcode != CL_SUCCESS) { printf("Error setting args for K2_q_col: %d\n", errcode); return; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2_q_col, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error launching K2_q_col: %d\n", errcode); return; }
}

// Launcher for kernel3_Rkj_partial_sum_gpu
// Returns the sum of partial sums from GPU work-groups.
DATA_TYPE launch_kernel3_Rkj_partial_sum_gpu(int k, int j_col, int m_rows_for_gpu, int n_total_cols, int local_size_x)
{
    if (m_rows_for_gpu <= 0) return 0.0;

    size_t localWorkSize[1] = {(size_t)local_size_x};
    size_t num_groups = (m_rows_for_gpu + local_size_x - 1) / local_size_x;
    if (num_groups > MAX_PARTIAL_SUM_GROUPS) num_groups = MAX_PARTIAL_SUM_GROUPS;
    if (num_groups == 0 && m_rows_for_gpu > 0) num_groups = 1;

    size_t globalWorkSize[1] = {num_groups * local_size_x};

    cl_mem partial_sum_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, num_groups * sizeof(DATA_TYPE), NULL, &errcode);
    if (errcode != CL_SUCCESS) { printf("Error creating buffer for partial sums K3_Rkj\n"); return 0.0; }

    errcode =  clSetKernelArg(clKernel3_Rkj_sum, 0, sizeof(cl_mem), (void *)&q_mem_obj);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 1, sizeof(cl_mem), (void *)&a_mem_obj);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 2, sizeof(int), (void *)&k);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 3, sizeof(int), (void *)&j_col);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 4, sizeof(int), (void *)&m_rows_for_gpu);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 5, sizeof(int), (void *)&n_total_cols);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 6, sizeof(cl_mem), (void *)&partial_sum_mem_obj);
    errcode |= clSetKernelArg(clKernel3_Rkj_sum, 7, local_size_x * sizeof(DATA_TYPE), NULL); // Local memory
    if(errcode != CL_SUCCESS) { printf("Error setting args for K3_Rkj_sum: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3_Rkj_sum, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error launching K3_Rkj_sum: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    DATA_TYPE partial_sums_host[MAX_PARTIAL_SUM_GROUPS];
    errcode = clEnqueueReadBuffer(clCommandQue, partial_sum_mem_obj, CL_TRUE, 0, num_groups * sizeof(DATA_TYPE), partial_sums_host, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error reading partial sums K3_Rkj: %d\n", errcode); clReleaseMemObject(partial_sum_mem_obj); return 0.0; }

    clReleaseMemObject(partial_sum_mem_obj);

    DATA_TYPE total_gpu_sum = 0.0;
    for (int i = 0; i < num_groups; i++) {
        total_gpu_sum += partial_sums_host[i];
    }
    return total_gpu_sum;
}

// Launcher for kernel3_update_Aij_gpu
void launch_kernel3_update_Aij_gpu(DATA_TYPE Rkj_value, int k, int j_col, int m_rows_for_gpu, int n_total_cols, int local_size_x)
{
    if (m_rows_for_gpu <= 0) return;

    size_t localWorkSize[1] = {(size_t)local_size_x};
    size_t globalWorkSize[1] = {(size_t)ceil(((float)m_rows_for_gpu) / local_size_x) * local_size_x};

    errcode =  clSetKernelArg(clKernel3_update_A, 0, sizeof(cl_mem), (void *)&a_mem_obj); // A is updated
    errcode |= clSetKernelArg(clKernel3_update_A, 1, sizeof(cl_mem), (void *)&q_mem_obj);
    errcode |= clSetKernelArg(clKernel3_update_A, 2, sizeof(DATA_TYPE), (void *)&Rkj_value);
    errcode |= clSetKernelArg(clKernel3_update_A, 3, sizeof(int), (void *)&k);
    errcode |= clSetKernelArg(clKernel3_update_A, 4, sizeof(int), (void *)&j_col);
    errcode |= clSetKernelArg(clKernel3_update_A, 5, sizeof(int), (void *)&m_rows_for_gpu);
    errcode |= clSetKernelArg(clKernel3_update_A, 6, sizeof(int), (void *)&n_total_cols);
    if(errcode != CL_SUCCESS) { printf("Error setting args for K3_update_A: %d\n", errcode); return; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3_update_A, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error launching K3_update_A: %d\n", errcode); return; }
}


// This is the cl_clean_up function that should remain.
// The old cl_launch_kernel and its associated cl_clean_up have been removed by previous steps or this one.
void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1_norm_sum);
	errcode = clReleaseKernel(clKernel2_q_col);
	errcode = clReleaseKernel(clKernel3_Rkj_sum);
	errcode = clReleaseKernel(clKernel3_update_A);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

// CPU helper functions
// Computes partial sum of squares for A[:,k] for rows m_start to m_end-1
DATA_TYPE cpu_norm_partial_sum(DATA_TYPE POLYBENCH_2D(A,M,N,m_total,n_total), int k, int m_start, int m_end)
{
    DATA_TYPE p_sum = 0.0;
    int i;
    for (i = m_start; i < m_end; i++) {
        p_sum += A[i][k] * A[i][k];
    }
    return p_sum;
}

// Computes Q[i][k] = A[i][k] / Rkk_val for rows m_start to m_end-1
void cpu_compute_q_col(DATA_TYPE POLYBENCH_2D(Q,M,N,m_total,n_total),
                         DATA_TYPE POLYBENCH_2D(A,M,N,m_total,n_total),
                         DATA_TYPE Rkk_val, int k, int m_start, int m_end)
{
    int i;
    for (i = m_start; i < m_end; i++) {
        Q[i][k] = A[i][k] / Rkk_val;
    }
}

// Computes partial sum for R[k][j] = Q[:,k] . A[:,j] for rows m_start to m_end-1
DATA_TYPE cpu_Rkj_partial_sum(DATA_TYPE POLYBENCH_2D(Q,M,N,m_total,n_total),
                                DATA_TYPE POLYBENCH_2D(A,M,N,m_total,n_total),
                                int k, int j_col, int m_start, int m_end)
{
    DATA_TYPE p_sum = 0.0;
    int i;
    for (i = m_start; i < m_end; i++) {
        p_sum += Q[i][k] * A[i][j_col];
    }
    return p_sum;
}

// Updates A[i][j_col] = A[i][j_col] - Q[i][k] * Rkj_val for rows m_start to m_end-1
void cpu_update_A_col(DATA_TYPE POLYBENCH_2D(A,M,N,m_total,n_total),
                        DATA_TYPE POLYBENCH_2D(Q,M,N,m_total,n_total),
                        DATA_TYPE Rkj_val, int k, int j_col, int m_start, int m_end)
{
    int i;
    for (i = m_start; i < m_end; i++) {
        A[i][j_col] = A[i][j_col] - Q[i][k] * Rkj_val;
    }
}


void gramschmidt_cpu_full(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(R,N,N,n,n), DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i][k] * A[i][k];
		}
		
		R[k][k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i][k] = A[i][k] / R[k][k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k][j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k][j] += Q[i][k] * A[i][j];
			}
			for (i = 0; i < M; i++)
			{
				A[i][j] = A[i][j] - Q[i][k] * R[k][j];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n, DATA_TYPE POLYBENCH_2D(A,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }

  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{	
    if (argc > 1) {
        alpha_split = atof(argv[1]);
        if (alpha_split < 0.0 || alpha_split > 1.0) {
            fprintf(stderr, "ERROR: Alpha value must be between 0.0 and 1.0.\n");
            return 1;
        }
    }

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,M,N,m,n); // For final GPU output comparison if needed
	POLYBENCH_2D_ARRAY_DECL(R_host,DATA_TYPE,N,N,n,n); // Host R matrix
	POLYBENCH_2D_ARRAY_DECL(Q_host,DATA_TYPE,M,N,m,n); // Host Q matrix

	init_array(POLYBENCH_ARRAY(A)); // Initialize A_host
	// Initialize R_host and Q_host to zeros
    int i_init, j_init;
    for (i_init = 0; i_init < N; i_init++) { // R is N x N
        for (j_init = 0; j_init < N; j_init++) {
            (*R_host)[i_init][j_init] = 0.0;
        }
    }
    for (i_init = 0; i_init < M; i_init++) { // Q is M x N
        for (j_init = 0; j_init < N; j_init++) {
            (*Q_host)[i_init][j_init] = 0.0;
        }
    }

	read_cl_file();
	cl_initialization();
	// Pass A, Q_host, R_host for initial device buffer population
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(Q_host), POLYBENCH_ARRAY(R_host));
	cl_load_prog();

    int m = M;
    int n = N;
    int k, j; // Loop iterators

    // Calculate row split for M dimension
    int gpu_m_rows = (int)(m * alpha_split);
    int cpu_m_rows = m - gpu_m_rows;
    int cpu_m_start = gpu_m_rows;

    printf("M=%d, N=%d\n", m, n);
    printf("GPU rows: 0 to %d (%d rows)\n", gpu_m_rows -1, gpu_m_rows);
    printf("CPU rows: %d to %d (%d rows)\n", cpu_m_start, m - 1, cpu_m_rows);

	/* Start timer. */
	polybench_start_instruments;

    // Main loop over k columns
    for (k = 0; k < n; k++)
    {
        // Step 1: Compute R[k][k] (norm of A[:,k])
        DATA_TYPE gpu_rkk_sum = 0.0;
        DATA_TYPE cpu_rkk_sum = 0.0;

        if (gpu_m_rows > 0) {
            // Ensure A[:,k] is up-to-date on device if modified in previous step 3b
            // This is handled by writing full A[:,j] back to device in step 3b.
            gpu_rkk_sum = launch_kernel1_norm_partial_sum_gpu(k, gpu_m_rows, n, DIM_LOCAL_WORK_GROUP_X);
        }
        if (cpu_m_rows > 0) {
            // A is A_host
            cpu_rkk_sum = cpu_norm_partial_sum(POLYBENCH_ARRAY(A), k, cpu_m_start, m);
        }
        clFinish(clCommandQue);

        (*R_host)[k][k] = sqrt(gpu_rkk_sum + cpu_rkk_sum);

        errcode = clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_FALSE, (k * n + k) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*R_host)[k][k], 0, NULL, NULL);
        if(errcode != CL_SUCCESS) { printf("Error writing R[k][k] to device, error %d\n", errcode); exit(1); }
        //clFinish(clCommandQue); // Finish can be delayed until data is needed by a kernel

        // Step 2: Compute Q[:,k] = A[:,k] / R[k][k]
        if (gpu_m_rows > 0) {
            // Kernel needs R[k][k] which can be passed as scalar arg. A[:,k] is on device.
            launch_kernel2_compute_q_col_gpu((*R_host)[k][k], k, gpu_m_rows, n, DIM_LOCAL_WORK_GROUP_X);
        }
        if (cpu_m_rows > 0) {
            cpu_compute_q_col(POLYBENCH_ARRAY(Q_host), POLYBENCH_ARRAY(A), (*R_host)[k][k], k, cpu_m_start, m);
        }

        // Read GPU part of Q[:,k] into Q_host
        if (gpu_m_rows > 0) {
            for(int row_idx = 0; row_idx < gpu_m_rows; ++row_idx) {
                 errcode = clEnqueueReadBuffer(clCommandQue, q_mem_obj, CL_FALSE, (row_idx * n + k) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*Q_host)[row_idx][k], 0, NULL, NULL);
                 if(errcode != CL_SUCCESS) { printf("Error reading Q[row=%d][k=%d] from GPU, error %d\n", row_idx, k, errcode); exit(1); }
            }
        }
        clFinish(clCommandQue); // Ensure Q_host is fully updated from GPU's computation.

        // Q_host[:,k] is now fully consistent.
        // Update q_mem_obj with the part of Q[:,k] that CPU computed.
        // GPU part is already in q_mem_obj.
        if (cpu_m_rows > 0) {
            for(int row_idx = cpu_m_start; row_idx < m; ++row_idx) {
                errcode = clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_FALSE, (row_idx * n + k) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*Q_host)[row_idx][k], 0, NULL, NULL);
                if(errcode != CL_SUCCESS) { printf("Error writing CPU part of Q[row=%d][k=%d] to GPU, error %d\n", row_idx, k, errcode); exit(1); }
            }
        }
        //clFinish(clCommandQue); // Ensure q_mem_obj is consistent. Delay if possible.


        // Step 3: For j from k+1 to N-1
        for (j = k + 1; j < n; j++)
        {
            clFinish(clCommandQue); // Ensure previous R[k][k] and Q[:,k] writes are done.

            // Step 3a: Compute R[k][j] = Q[:,k] . A[:,j]
            DATA_TYPE gpu_rkj_sum = 0.0;
            DATA_TYPE cpu_rkj_sum = 0.0;

            if (gpu_m_rows > 0) {
                // Q[:,k] is on q_mem_obj. A[:,j] is on a_mem_obj.
                gpu_rkj_sum = launch_kernel3_Rkj_partial_sum_gpu(k, j, gpu_m_rows, n, DIM_LOCAL_WORK_GROUP_X);
            }
            if (cpu_m_rows > 0) {
                // Q_host has full Q[:,k], A is A_host for A[:,j]
                cpu_rkj_sum = cpu_Rkj_partial_sum(POLYBENCH_ARRAY(Q_host), POLYBENCH_ARRAY(A), k, j, cpu_m_start, m);
            }
            clFinish(clCommandQue);

            (*R_host)[k][j] = gpu_rkj_sum + cpu_rkj_sum;

            errcode = clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_FALSE, (k * n + j) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*R_host)[k][j], 0, NULL, NULL);
            if(errcode != CL_SUCCESS) { printf("Error writing R[k][j=%d] to device, error %d\n",j, errcode); exit(1); }
            //clFinish(clCommandQue);


            // Step 3b: Update A[:,j] = A[:,j] - Q[:,k] * R[k][j]
            if (gpu_m_rows > 0) {
                // Q[:,k] on q_mem_obj. R[k][j] passed as scalar. A[:,j] on a_mem_obj.
                launch_kernel3_update_Aij_gpu((*R_host)[k][j], k, j, gpu_m_rows, n, DIM_LOCAL_WORK_GROUP_X);
            }
            if (cpu_m_rows > 0) {
                // Q_host has Q[:,k]. R_host[k][j] is scalar. A is A_host.
                cpu_update_A_col(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(Q_host), (*R_host)[k][j], k, j, cpu_m_start, m);
            }

            // Read GPU part of A[:,j] into A_host (which has CPU part)
            if (gpu_m_rows > 0) {
                 for(int row_idx = 0; row_idx < gpu_m_rows; ++row_idx) {
                    errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_FALSE, (row_idx * n + j) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*A)[row_idx][j], 0, NULL, NULL);
                    if(errcode != CL_SUCCESS) { printf("Error reading A[row=%d][j=%d] from GPU, error %d\n",row_idx,j, errcode); exit(1); }
                 }
            }
            clFinish(clCommandQue); // Ensure A_host is updated from GPU's computation.

            // A_host[:,j] is now fully consistent.
            // Update a_mem_obj with the part of A[:,j] that CPU computed.
            // GPU part is already in a_mem_obj.
            if (cpu_m_rows > 0) {
                for(int row_idx = cpu_m_start; row_idx < m; ++row_idx) {
                     errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_FALSE, (row_idx * n + j) * sizeof(DATA_TYPE), sizeof(DATA_TYPE), &(*A)[row_idx][j], 0, NULL, NULL);
                     if(errcode != CL_SUCCESS) { printf("Error writing CPU part of A[row=%d][j=%d] to GPU, error %d\n",row_idx,j, errcode); exit(1); }
                }
            }
            //clFinish(clCommandQue); // Ensure a_mem_obj is consistent. Delay if possible.
        }
        clFinish(clCommandQue); // Finish all ops for column k (all j's processed for this k)
    }

	/* Stop and print timer. */
	printf("CPU-GPU Combined Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

    // Read final A matrix from GPU to A_outputFromGpu for comparison
	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, M*N*sizeof(DATA_TYPE), POLYBENCH_ARRAY(A_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading final A from GPU to A_outputFromGpu\n");
    clFinish(clCommandQue);

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		// A_host is used as input, R_host and Q_host are outputs for the CPU full version
		gramschmidt_cpu_full(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R_host), POLYBENCH_ARRAY(Q_host));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
        // After cl_launch_kernel, A_outputFromGpu will hold the result from combined CPU-GPU run.
        // A (host) here is the original A modified by gramschmidt_cpu_full.
		compareResults(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(M, N, POLYBENCH_ARRAY(A_outputFromGpu)); // Print the A from combined run

	#endif //RUN_ON_CPU


	cl_clean_up();

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);
	POLYBENCH_FREE_ARRAY(R_host);
	POLYBENCH_FREE_ARRAY(Q_host);

	return 0;
}

#include "../../common/polybench.c"
