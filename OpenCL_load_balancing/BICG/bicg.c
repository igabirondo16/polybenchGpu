/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "bicg.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Alpha ( доля работы для GPU: 0.0 = только CPU, 1.0 = только GPU) будет передаваться как аргумент командной строки.
// Default value is 0.5 if not provided.
// Example: #define ALPHA 0.7 // 70% on GPU, 30% on CPU

#define MAX_SOURCE_SIZE (0x100000)

#ifndef M_PI
#define M_PI 3.14159
#endif

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
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem p_mem_obj;
cl_mem q_mem_obj;
cl_mem s_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
		DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<nx; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<ny; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("bicg.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	int i, j;
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}

	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NX;
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


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny),
	DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	p_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NY, NULL, &errcode);

	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	s_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, sizeof(DATA_TYPE) * NY, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
	
	// Write data to read-only buffers
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, A, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, r, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, p_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NY, p, 0, NULL, NULL);

	// Initialize output buffers q and s (optional, as they are write-only, but good practice if debugging or if kernels did read-modify-write)
	// For now, we assume kernels fully overwrite the portions they are responsible for.
	// DATA_TYPE* zero_nx = (DATA_TYPE*)calloc(NX, sizeof(DATA_TYPE));
	// DATA_TYPE* zero_ny = (DATA_TYPE*)calloc(NY, sizeof(DATA_TYPE));
	// errcode |= clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, zero_nx, 0, NULL, NULL);
	// errcode |= clEnqueueWriteBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NY, zero_ny, 0, NULL, NULL);
	// free(zero_nx);
	// free(zero_ny);

	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
 }

void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the 1st OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "bicgKernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	// Create the 2nd OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "bicgKernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	clFinish(clCommandQue);
}

// Launches bicgKernel1: q = A * p
// nx_total and ny_total are the full dimensions of A (_PB_NX, _PB_NY)
// gpu_elements_q is the number of elements of q the GPU should compute (gpu_nx from main)
void cl_launch_bicg_kernel1_gpu(int nx_total, int ny_total, int gpu_elements_q)
{
	if (gpu_elements_q <= 0) return;

	size_t localWorkSize[2];
	size_t globalWorkSize[2];

	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)gpu_elements_q) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&p_mem_obj); // Kernel 1 uses p
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), &nx_total); // Full NX
    errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), &ny_total); // Full NY
	if(errcode != CL_SUCCESS) printf("Error in setting arguments for Kernel 1\n");

	// Execute the 1st OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching Kernel 1\n");
	// clFinish is called outside by main orchestrator
}

// Launches bicgKernel2: s = A^T * r
// nx_total and ny_total are the full dimensions of A (_PB_NX, _PB_NY)
// gpu_elements_s is the number of elements of s the GPU should compute (gpu_ny from main)
void cl_launch_bicg_kernel2_gpu(int nx_total, int ny_total, int gpu_elements_s)
{
	if (gpu_elements_s <= 0) return;

	size_t localWorkSize[2];
	size_t globalWorkSize[2];

	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)gpu_elements_s) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj); // Kernel 2 uses r
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&s_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), &nx_total); // Full NX
    errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), &ny_total); // Full NY
	if(errcode != CL_SUCCESS) printf("Error in setting arguments for Kernel 2\n");
	
	// Execute the 2nd OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching Kernel 2\n");
	// clFinish is called outside by main orchestrator
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(p_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(s_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

// CPU helper function for bicgKernel1: q = A * p
// Computes q[i] for i in [start_row_q, end_row_q)
void bicg_cpu_kernel1_partial(DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny_orig), DATA_TYPE POLYBENCH_1D(p,NY,ny_orig),
                              DATA_TYPE POLYBENCH_1D(q,NX,nx_orig), int ny, int start_row_q, int end_row_q)
{
    int i, j;
    for (i = start_row_q; i < end_row_q; i++)
    {
        q[i] = 0.0;
        for (j = 0; j < ny; j++) // ny is _PB_NY (total columns in A, total elements in p)
        {
            q[i] += A[i][j] * p[j];
        }
    }
}

// CPU helper function for bicgKernel2: s = A^T * r
// Computes s[j] for j in [start_col_s, end_col_s)
void bicg_cpu_kernel2_partial(DATA_TYPE POLYBENCH_2D(A,NX,NY,nx_orig,ny_orig), DATA_TYPE POLYBENCH_1D(r,NX,nx_orig),
                              DATA_TYPE POLYBENCH_1D(s,NY,ny_orig), int nx, int start_col_s, int end_col_s)
{
    int i, j;
    for (j = start_col_s; j < end_col_s; j++)
    {
        s[j] = 0.0;
        for (i = 0; i < nx; i++) // nx is _PB_NX (total rows in A, total elements in r)
        {
            s[j] += A[i][j] * r[i];
        }
    }
}


// Original bicg_cpu for reference or full CPU run
void bicg_cpu_full(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx),
              DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;
	
	// s = A^T * r
	for (i = 0; i < ny; i++) // Iterate over columns of A / elements of s
	{
		s[i] = 0.0;
		for (j = 0; j < nx; j++) // Iterate over rows of A / elements of r
		{
			s[i] += A[j][i] * r[j]; // A[j][i] for transpose
		}
	}

	// q = A * p
	for (i = 0; i < nx; i++) // Iterate over rows of A / elements of q
	{
		q[i] = 0.0;
		for (j = 0; j < ny; j++) // Iterate over columns of A / elements of p
	  	{
	    		q[i] = q[i] + A[i][j] * p[j];
	  	}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, int ny,
		 DATA_TYPE POLYBENCH_1D(s,NY,ny),
		 DATA_TYPE POLYBENCH_1D(q,NX,nx))

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[]) 
{
	double alpha = 0.5; // Default alpha
	if (argc > 1) {
	alpha = atof(argv[1]);
	if (alpha < 0.0 || alpha > 1.0) {
		fprintf(stderr, "Alpha must be between 0.0 and 1.0\n");
		return 1;
	}
	}

	int nx = NX;
	int ny = NY;
	
	// Alpha (alpha_gpu_share) defines the proportion of work done by the GPU.
	// GPU handles the first 'alpha_gpu_share * total_elements'
	// CPU handles the remaining '(1 - alpha_gpu_share) * total_elements'
	double alpha_cpu_share = 0.5; // Default alpha: 50% for GPU, 50% for CPU
	if (argc > 1) {
		alpha_cpu_share = atof(argv[1]);
		if (alpha_cpu_share < 0.0 || alpha_cpu_share > 1.0) {
			fprintf(stderr, "Alpha (CPU share) must be between 0.0 and 1.0\n");
			return 1;
		}
	}


	double alpha_gpu_share = 1.0f - alpha_cpu_share;
	// Calculate split points based on alpha_gpu_share
	// For Kernel 1 (bicgKernel1: q = A * p, computation over NX elements of q)
	// GPU computes the first 'gpu_elements_q' elements of q.
	// CPU computes the remaining 'cpu_elements_q' elements of q.
	int gpu_elements_q = (int)(nx * alpha_gpu_share);
	int cpu_elements_q_start_index = gpu_elements_q;
	int cpu_elements_q = nx - gpu_elements_q;

	// For Kernel 2 (bicgKernel2: s = A^T * r, computation over NY elements of s)
	// GPU computes the first 'gpu_elements_s' elements of s.
	// CPU computes the remaining 'cpu_elements_s' elements of s.
	int gpu_elements_s = (int)(ny * alpha_gpu_share);
	int cpu_elements_s_start_index = gpu_elements_s;
	int cpu_elements_s = ny - gpu_elements_s;

	printf("BICG Benchmark with CPU-GPU Collaboration\n");
	printf("Alpha (CPU Share): %.2f\n", alpha_cpu_share);
	printf("NX: %d, NY: %d\n\n", nx, ny);

	printf("Kernel 1 (q = A * p, dimension NX = %d):\n", nx);
	printf("  GPU computes first %d elements of q (0 to %d)\n", gpu_elements_q, gpu_elements_q - 1);
	printf("  CPU computes last %d elements of q (%d to %d)\n\n", cpu_elements_q, cpu_elements_q_start_index, nx - 1);

	printf("Kernel 2 (s = A^T * r, dimension NY = %d):\n", ny);
	printf("  GPU computes first %d elements of s (0 to %d)\n", gpu_elements_s, gpu_elements_s - 1);
	printf("  CPU computes last %d elements of s (%d to %d)\n\n", cpu_elements_s, cpu_elements_s_start_index, ny - 1);

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(s_combined,DATA_TYPE,NY,ny); // s_combined will store the final s
	POLYBENCH_1D_ARRAY_DECL(q_combined,DATA_TYPE,NX,nx); // q_combined will store the final q
	POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	//POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny); // Not strictly needed if reading directly
	//POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx); // Not strictly needed if reading directly

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));
	
	read_cl_file();
	cl_initialization();
	// Pass s_combined and q_combined to cl_mem_init for initial setup, though they are written by kernels/CPU
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s_combined), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q_combined));
	cl_load_prog();

	/* Start timer. */
	polybench_start_instruments;

	// --- Kernel 1: q = A * p ---
	// GPU part for q
	if (gpu_elements_q > 0) {
		cl_launch_bicg_kernel1_gpu(nx, ny, gpu_elements_q);
	}
	// CPU part for q
	if (cpu_elements_q > 0) {
		bicg_cpu_kernel1_partial(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q_combined),
		                         ny, cpu_elements_q_start_index, nx);
	}
	// Read back GPU part of q into the beginning of q_combined
	if (gpu_elements_q > 0) {
		errcode = clEnqueueReadBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, gpu_elements_q * sizeof(DATA_TYPE),
		                              POLYBENCH_ARRAY(q_combined), 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading q_mem_obj from GPU (Kernel 1)\n");
	}
    clFinish(clCommandQue); // Synchronize after Kernel 1 and its data transfers

	// --- Kernel 2: s = A^T * r ---
	// GPU part for s
	if (gpu_elements_s > 0) {
		cl_launch_bicg_kernel2_gpu(nx, ny, gpu_elements_s);
	}
	// CPU part for s
	if (cpu_elements_s > 0) {
		bicg_cpu_kernel2_partial(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s_combined),
		                         nx, cpu_elements_s_start_index, ny);
	}
	// Read back GPU part of s into the beginning of s_combined
	if (gpu_elements_s > 0) {
		errcode = clEnqueueReadBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, gpu_elements_s * sizeof(DATA_TYPE),
		                              POLYBENCH_ARRAY(s_combined), 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading s_mem_obj from GPU (Kernel 2)\n");
	}
	clFinish(clCommandQue); // Synchronize after Kernel 2 and its data transfers

	/* Stop and print timer. */
	printf("\nCPU-GPU Time in seconds: ");
	polybench_stop_instruments;
	polybench_print_instruments;

	size_t a_size = sizeof(DATA_TYPE) * NX * NY;
	size_t other_buffers = 4 * sizeof(DATA_TYPE) * NX;
	
	size_t buffer_size = a_size + other_buffers;
	size_t arg_size = 2*sizeof(int);

	size_t total_bytes = buffer_size + arg_size;
	printf("Total bytes: %ld\n", total_bytes);

	size_t wg_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
	printf("Work group size: %ld\n", wg_size);

	#ifdef RUN_ON_CPU
		/* Start timer. */
	  	polybench_start_instruments;

		// Run full computation on CPU for comparison
		POLYBENCH_1D_ARRAY_DECL(s_cpu_ref,DATA_TYPE,NY,ny);
		POLYBENCH_1D_ARRAY_DECL(q_cpu_ref,DATA_TYPE,NX,nx);
		
		bicg_cpu_full(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s_cpu_ref), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q_cpu_ref));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(nx, ny, POLYBENCH_ARRAY(s_cpu_ref), POLYBENCH_ARRAY(s_combined), POLYBENCH_ARRAY(q_cpu_ref), POLYBENCH_ARRAY(q_combined));
		
		POLYBENCH_FREE_ARRAY(s_cpu_ref);
		POLYBENCH_FREE_ARRAY(q_cpu_ref);
	#else
		print_array(nx, ny, POLYBENCH_ARRAY(s_combined), POLYBENCH_ARRAY(q_combined));
	#endif

	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s_combined);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q_combined);
	//POLYBENCH_FREE_ARRAY(s_outputFromGpu); // Not used
	//POLYBENCH_FREE_ARRAY(q_outputFromGpu); // Not used
	
    	return 0;
}

#include "../../common/polybench.c"
