/**
 * 2DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "2DConvolution.h"

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
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU


void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
	int i, j, fail;
	fail = 0;

	// Compare outputs from CPU and GPU
	for (i=1; i < (ni-1); i++)
	{
		for (j=1; j < (nj-1); j++)
		{
			if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
	fp = fopen("2DConvolution.cl", "r");
	if (!fp) {
		fprintf(stdout, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
	int i, j;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			A[i][j] = (float)rand()/RAND_MAX;
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


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, A, 0, NULL, NULL);
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
	clKernel = clCreateKernel(clProgram, "Convolution2D_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	clFinish(clCommandQue);
}


void cl_launch_kernel(int start_row, int num_rows, int ni, int nj)
{
	size_t localWorkSize[2], globalWorkSize[2];

	// Set workgroup sizes
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;

	// Calculate global work size (make sure it's a multiple of local size)
	// For rows (dimension 0), we only process num_rows rows
	globalWorkSize[0] = (size_t)ceil(((float)num_rows) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	// For columns (dimension 1), we process all nj columns
	globalWorkSize[1] = (size_t)ceil(((float)nj) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	printf("NI = %d, num_rows = %d, start_row = %d\n", ni, num_rows, start_row);
	printf("Global work size: [%zu, %zu], Local work size: [%zu, %zu]\n",
		globalWorkSize[0], globalWorkSize[1], localWorkSize[0], localWorkSize[1]);

	// Ensure start_row is valid
	if (start_row >= ni) {
		printf("Error: start_row (%d) is >= ni (%d)\n", start_row, ni);
		return;
	}

	// Ensure we don't exceed array bounds
	if (start_row + num_rows > ni) {
		printf("Error: start_row (%d) + num_rows (%d) exceeds ni (%d)\n",
			start_row, num_rows, ni);
		return;
	}

	/* Start timer. */
  	polybench_start_instruments;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode =  clSetKernelArg(clKernel, 2, sizeof(int), &ni);
	errcode |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);
	errcode |= clSetKernelArg(clKernel, 4, sizeof(int), &start_row);

	if(errcode != CL_SUCCESS) {
		printf("Error in setting arguments: %d\n", errcode);
		return;
	}

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);

	if(errcode != CL_SUCCESS) {
		printf("Error in launching kernel: %d\n", errcode);
		return;
	}

	// Make sure kernel finishes before we continue
	clFinish(clCommandQue);

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
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

void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
		for (j = 1; j < _PB_NJ - 1; ++j) // 1
		{
			B[i][j] = c11 * A[(i - 1)][(j - 1)]  +  c12 * A[(i + 0)][(j - 1)]  +  c13 * A[(i + 1)][(j - 1)]
				+ c21 * A[(i - 1)][(j + 0)]  +  c22 * A[(i + 0)][(j + 0)]  +  c23 * A[(i + 1)][(j + 0)]
				+ c31 * A[(i - 1)][(j + 1)]  +  c32 * A[(i + 0)][(j + 1)]  +  c33 * A[(i + 1)][(j + 1)];
		}
	}
}

void conv2D_partial(int start_row, int end_row, int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = start_row; i < end_row; ++i) // 0
	{
		for (j = 1; j < nj - 1; ++j) // 1
		{
			if (i >= 1 && j >= 1 && i < ni - 1 && j < nj - 1)
			{
				B[i][j] = c11 * A[(i - 1)][(j - 1)]  +  c12 * A[(i + 0)][(j - 1)]  +  c13 * A[(i + 1)][(j - 1)]
					+ c21 * A[(i - 1)][(j + 0)]  +  c22 * A[(i + 0)][(j + 0)]  +  c23 * A[(i + 1)][(j + 0)]
					+ c31 * A[(i - 1)][(j + 1)]  +  c32 * A[(i + 0)][(j + 1)]  +  c33 * A[(i + 1)][(j + 1)];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	float alpha = 0.5f;
	if (argc > 1) {
		alpha = atof(argv[1]);
	}

	/* Retrieve problem size */
	int ni = NI;
	int nj = NJ;

	// Calculate work distribution between CPU and GPU
	int cpu_start = (int)((1.0f - alpha) * ni);
	int cpu_end = ni;
	int cpu_rows = cpu_end - cpu_start;

	int gpu_start = 0;
	int gpu_end = cpu_start;
	int gpu_rows = gpu_end - gpu_start;


	//int cpu_start = 0;
	//int cpu_end = (int)(alpha * (ni));
	//int cpu_rows = cpu_end - cpu_start;

	//int gpu_start = cpu_end;
	//int gpu_end = ni;
	//int gpu_rows = gpu_end - gpu_start;

	// Validate work distribution
	if (gpu_start >= ni && alpha < 1) {
		printf("Invalid work distribution: gpu_start (%d) >= ni (%d)\n",
		       gpu_start, ni);
		return 1;
	}

	printf("cpu_start = %d, cpu_end = %d, cpu_rows = %d\n", cpu_start, cpu_end, cpu_rows);
	printf("gpu_start = %d, gpu_end = %d, gpu_rows = %d\n", gpu_start, gpu_end, gpu_rows);
	printf("\n");

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

	// Initialize arrays with random data
	init(ni, nj, POLYBENCH_ARRAY(A));

	// Initialize B_outputFromGpu to known values to avoid garbage data
	//for (int i = 0; i < ni; i++) {
	//	for (int j = 0; j < nj; j++) {
	//		B_outputFromGpu[i][j] = 0.0f;
	//	}
	//}

	// Setup OpenCL environment
	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A));
	cl_load_prog();

	// Execute CPU portion if any rows are assigned to CPU
	if (cpu_rows > 0) {
		printf("Running CPU computation for rows %d to %d...\n", cpu_start, cpu_end-1);
		conv2D_partial(cpu_start, cpu_end, ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B_outputFromGpu));
	}

	// Execute GPU portion if any rows are assigned to GPU
	if (gpu_rows > 0) {
		printf("Running GPU computation for rows %d to %d...\n", gpu_start, gpu_end-1);
		cl_launch_kernel(gpu_start, gpu_rows, ni, nj);

		// Make sure GPU is done
		clFinish(clCommandQue);

		// Calculate memory sizes in elements, not bytes
		size_t element_size = sizeof(DATA_TYPE);
		size_t elements_per_row = NJ;
		//size_t offset_elements = gpu_start * elements_per_row;
		size_t offset_elements = 0;

		size_t offset_bytes = offset_elements * element_size;
		size_t num_elements_to_read = gpu_rows * elements_per_row;
		size_t bytes_to_read = num_elements_to_read * element_size;

		// Validate that we're not reading beyond buffer bounds
		size_t buffer_size = NI * NJ * element_size;
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

	#ifdef RUN_ON_CPU

		/* Start timer. */
  		polybench_start_instruments;

		conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
 		polybench_print_instruments;

		compareResults(ni, nj, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nj, POLYBENCH_ARRAY(B_outputFromGpu));

	#endif //RUN_ON_CPU


	cl_clean_up();

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    	return 0;
}

#include "../../common/polybench.c"

