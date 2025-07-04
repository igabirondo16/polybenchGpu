/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "mvt.h"
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
cl_mem x1_mem_obj;
cl_mem x2_mem_obj;
cl_mem y1_mem_obj;
cl_mem y2_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;
const int LIST_SIZE = N;
char str_temp[1024];

#define RUN_ON_CPU


void compareResults(int n, DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
	int i, fail;
	fail = 0;

	for (i=0; i<n; i++)
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("mvt.cl", "r");
	if (!fp) {
		fprintf(stdout, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
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


void cl_mem_init(DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_1D(x1,N,n), DATA_TYPE POLYBENCH_1D(x2,N,n),
	DATA_TYPE POLYBENCH_1D(y_1,N,n), DATA_TYPE POLYBENCH_1D(y_2,N,n))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	x1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	x2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, a, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_2, 0, NULL, NULL);

	if(errcode != CL_SUCCESS) printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program %d\n",errcode);

	// Create the 1st OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "mvt_kernel1", &errcode);
	// Create the 2nd OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "mvt_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int n, int gpu_rows)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)gpu_rows) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&y1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&x2_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&y2_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");

	clFinish(clCommandQue);
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
	errcode = clReleaseMemObject(x1_mem_obj);
	errcode = clReleaseMemObject(x2_mem_obj);
	errcode = clReleaseMemObject(y1_mem_obj);
	errcode = clReleaseMemObject(y2_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void runMvt(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
	int i, j;

	for (i=0; i<_PB_N; i++)
	{
		for (j=0; j<N; j++)
		{
       		x1[i] = x1[i] + a[i][j] * y1[j];
		}
    }

	for (i=0; i<_PB_N; i++)
	{
		for (j=0; j<_PB_N; j++)
		{
 		      	x2[i] = x2[i] + a[j][i] * y2[j];
      		}
    	}
}

void runMvt_partial(
	int n,
	int start_row,
	int end_row,
	DATA_TYPE POLYBENCH_2D(a, N, N, n, n),
	DATA_TYPE POLYBENCH_1D(x1, N, n),
	DATA_TYPE POLYBENCH_1D(x2, N, n),
	DATA_TYPE POLYBENCH_1D(y1, N, n),
	DATA_TYPE POLYBENCH_1D(y2, N, n)
)
{
	int i, j;

	for (i=start_row; i<end_row; i++)
	{
		for (j=0; j<N; j++)
		{
       		x1[i] = x1[i] + a[i][j] * y1[j];
		}
    }

	for (i=start_row; i<end_row; i++)
	{
		for (j=0; j<_PB_N; j++)
		{
 		      	x2[i] = x2[i] + a[j][i] * y2[j];
      		}
    	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, x1[i]);
    fprintf (stderr, DATA_PRINTF_MODIFIER, x2[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}



int main(int argc, char *argv[])
{
	// Input alpha parameter for splitting the work
	float alpha = 0.5f;
	if (argc > 1) {
		alpha = atof(argv[1]);
	}

	int n = N;

	// Calculate work distribution between CPU and GPU
	int cpu_start = (int)((1.0f - alpha) * n);
	int cpu_end = n;
	int cpu_rows = cpu_end - cpu_start;

	int gpu_start = 0;
	int gpu_end = cpu_start;
	int gpu_rows = gpu_end - gpu_start;

	printf("cpu_start = %d, cpu_end = %d, cpu_rows = %d\n", cpu_start, cpu_end, cpu_rows);
	printf("gpu_start = %d, gpu_end = %d, gpu_rows = %d\n", gpu_start, gpu_end, gpu_rows);
	printf("\n");

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_1D_ARRAY_DECL(x1,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x2,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x2_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_1,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_2,DATA_TYPE,N,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));
	cl_load_prog();

	/* Start timer. */
  	polybench_start_instruments;

	if (cpu_rows > 0) {
		runMvt_partial(n, cpu_start, cpu_end, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));
	}

	if (gpu_rows > 0) {
		cl_launch_kernel(n, gpu_rows);

		errcode = clEnqueueReadBuffer(clCommandQue, x1_mem_obj, CL_TRUE, 0, gpu_rows*sizeof(DATA_TYPE), POLYBENCH_ARRAY(x1_outputFromGpu), 0, NULL, NULL);
		errcode = clEnqueueReadBuffer(clCommandQue, x2_mem_obj, CL_TRUE, 0, gpu_rows*sizeof(DATA_TYPE), POLYBENCH_ARRAY(x2_outputFromGpu), 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	}

	/* Stop and print timer. */
	printf("\nCPU-GPU Time in seconds: ");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	size_t a_size = sizeof(DATA_TYPE) * N * N;
	size_t other_size = sizeof(DATA_TYPE) * N * 4;
	size_t buffer_size = a_size + other_size;
	size_t arg_size = sizeof(int);

	size_t total_bytes = buffer_size + arg_size;
	printf("Total bytes: %ld\n", total_bytes); 

	size_t wg_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
	printf("Work group size: %ld\n", wg_size);

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runMvt(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(x2_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(n, POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));

	#endif //RUN_ON_CPU


	cl_clean_up();

	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(x1);
	POLYBENCH_FREE_ARRAY(x2);
	POLYBENCH_FREE_ARRAY(x1_outputFromGpu);
	POLYBENCH_FREE_ARRAY(x2_outputFromGpu);
	POLYBENCH_FREE_ARRAY(y_1);
	POLYBENCH_FREE_ARRAY(y_2);

	return 0;
}

#include "../../common/polybench.c"
