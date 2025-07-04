/**
 * 3DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3DConvolution.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

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
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("3DConvolution.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			for (k = 0; k < nk; ++k)
			{
				A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
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


void cl_mem_init(DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, B, 0, NULL, NULL);
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

	// Create the OpenCL kernel
	clKernel = clCreateKernel(clProgram, "Convolution3D_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int gpu_num_rows, int ni, int nj, int nk)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NK) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel, 2, sizeof(int), &ni);
	errcode |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);
	errcode |= clSetKernelArg(clKernel, 4, sizeof(int), &nk);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	int i;
	for (i = 0; i < gpu_num_rows; ++i) // 0
	{
		// set the current value of 'i' for the argument in the kernel
		errcode |= clSetKernelArg(clKernel, 5, sizeof(int), &i);

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	}

	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void compareResults(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k, fail;
	fail = 0;

	// Compare result from cpu and gpu
	for (i = 1; i < ni - 1; ++i) // 0
	{
		for (j = 1; j < nj - 1; ++j) // 1
		{
			for (k = 1; k < nk - 1; ++k) // 2
			{
				if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void conv3D(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
		for (j = 1; j < _PB_NJ - 1; ++j) // 1
		{
			for (k = 1; k < _PB_NK -1; ++k) // 2
			{
				B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)]  +  c13 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c21 * A[(i - 1)][(j - 1)][(k - 1)]  +  c23 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c31 * A[(i - 1)][(j - 1)][(k - 1)]  +  c33 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c12 * A[(i + 0)][(j - 1)][(k + 0)]  +  c22 * A[(i + 0)][(j + 0)][(k + 0)]
					     +   c32 * A[(i + 0)][(j + 1)][(k + 0)]  +  c11 * A[(i - 1)][(j - 1)][(k + 1)]
					     +   c13 * A[(i + 1)][(j - 1)][(k + 1)]  +  c21 * A[(i - 1)][(j + 0)][(k + 1)]
					     +   c23 * A[(i + 1)][(j + 0)][(k + 1)]  +  c31 * A[(i - 1)][(j + 1)][(k + 1)]
					     +   c33 * A[(i + 1)][(j + 1)][(k + 1)];
			}
		}
	}
}

void conv3D_partial(int start_row, int end_row, int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = start_row; i < end_row; ++i) // 0
	{
		for (j = 1; j < _PB_NJ - 1; ++j) // 1
		{
			for (k = 1; k < _PB_NK -1; ++k) // 2
			{
				if ((i < (ni - 1)) && (j < (nj - 1)) && (k < (nk - 1)) && (i > 0) && (j > 0) && (k > 0))
				{
					B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)]  +  c13 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c21 * A[(i - 1)][(j - 1)][(k - 1)]  +  c23 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c31 * A[(i - 1)][(j - 1)][(k - 1)]  +  c33 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c12 * A[(i + 0)][(j - 1)][(k + 0)]  +  c22 * A[(i + 0)][(j + 0)][(k + 0)]
					     +   c32 * A[(i + 0)][(j + 1)][(k + 0)]  +  c11 * A[(i - 1)][(j - 1)][(k + 1)]
					     +   c13 * A[(i + 1)][(j - 1)][(k + 1)]  +  c21 * A[(i - 1)][(j + 0)][(k + 1)]
					     +   c23 * A[(i + 1)][(j + 0)][(k + 1)]  +  c31 * A[(i - 1)][(j + 1)][(k + 1)]
					     +   c33 * A[(i + 1)][(j + 1)][(k + 1)];
				}
			}
		}
	}
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, int nk,
		 DATA_TYPE POLYBENCH_3D(B,NI,NJ,NK,ni,nj,nk))
{
  int i, j, k;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
	for (k = 0; k < nk; k++)
	{
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
	if ((i * (nj*nk) + j*nk + k) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	// Input alpha parameter for splitting the work
	float alpha = 0.5f;
	if (argc > 1) {
		alpha = atof(argv[1]);
	}

	// Work dimensions
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	// Calculate work distribution between CPU and GPU
	int cpu_start = (int)((1.0f - alpha) * ni);
	int cpu_end = ni;
	int cpu_rows = cpu_end - cpu_start;

	int gpu_start = 0;
	int gpu_end = cpu_start;
	int gpu_rows = gpu_end - gpu_start;

	printf("cpu_start = %d, cpu_end = %d, cpu_rows = %d\n", cpu_start, cpu_end, cpu_rows);
	printf("gpu_start = %d, gpu_end = %d, gpu_rows = %d\n", gpu_start, gpu_end, gpu_rows);
	printf("\n");

	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,NK,ni,nj,nk);

	init(ni, nj, nk, POLYBENCH_ARRAY(A));

	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
	cl_load_prog();

	/* Start timer. */
  	polybench_start_instruments;

	if (cpu_rows > 0) {
		printf("Running CPU computation for rows %d to %d...\n", cpu_start, cpu_end);
		conv3D_partial(cpu_start, cpu_end, ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B_outputFromGpu));
	}

	// Execute GPU portion if any rows are assigned to GPU
	if (gpu_rows > 0) {
		printf("Running GPU computation for rows %d to %d...\n", gpu_start, gpu_end);
		cl_launch_kernel(gpu_rows, ni, nj, nk);

		// Make sure GPU is done
		clFinish(clCommandQue);

		// Calculate memory sizes in elements, not bytes
		size_t element_size = sizeof(DATA_TYPE);
		size_t elements_per_row = NJ * NK;
		//size_t offset_elements = gpu_start * elements_per_row;
		size_t offset_elements = 0;

		size_t offset_bytes = offset_elements * element_size;
		size_t num_elements_to_read = gpu_rows * elements_per_row;
		size_t bytes_to_read = num_elements_to_read * element_size;

		// Validate that we're not reading beyond buffer bounds
		size_t buffer_size = NI * NJ * NK * element_size;
		if (offset_bytes + bytes_to_read > buffer_size) {
			printf("Error: Attempting to read beyond buffer bounds.\n");
			printf("  Buffer size: %zu bytes\n", buffer_size);
			printf("  Offset: %zu bytes\n", offset_bytes);
			printf("  Read size: %zu bytes\n", bytes_to_read);
			printf("  Exceeds by: %zu bytes\n", offset_bytes + bytes_to_read - buffer_size);
			return 1;
		}

		printf("Reading GPU results - gpu_start: %d, gpu_end: %d, offset: %zu, read_size: %zu\n",
		       gpu_start, gpu_end, offset_bytes, bytes_to_read);

		// Read results from GPU memory
		errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, offset_bytes, bytes_to_read,
		                             B_outputFromGpu, 0, NULL, NULL);

		if(errcode != CL_SUCCESS) {
			printf("Error in reading GPU mem: %d\n", errcode);
			return 1;
		}

		// Ensure all reads are complete
		clFinish(clCommandQue);
	}

	/* Stop and print timer. */
	printf("\nCPU-GPU Time in seconds: ");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	size_t arg_size = sizeof(DATA_TYPE) * NI * NJ * NK;
	size_t buffer_size = sizeof(int) * 3;

	size_t total_bytes = buffer_size + arg_size;
	printf("Total bytes: %ld\n", total_bytes);

	size_t wg_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
	printf("Work group size: %ld\n", wg_size);

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nj, nk, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nj, nk, POLYBENCH_ARRAY(B_outputFromGpu));

	#endif //RUN_ON_CPU

	cl_clean_up();

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"
