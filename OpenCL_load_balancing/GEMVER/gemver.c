/**
 * gemver.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemver.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#define RUN_ON_CPU

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

DATA_TYPE ALPHA = 23;
DATA_TYPE BETA = 15;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem z_mem_obj;
cl_mem v1_mem_obj;
cl_mem v2_mem_obj;
cl_mem u1_mem_obj;
cl_mem u2_mem_obj;
cl_mem w_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

//#define RUN_ON_CPU


void compareResults(DATA_TYPE POLYBENCH_1D(w1,N,n), DATA_TYPE POLYBENCH_1D(w2,N,n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	fp = fopen("gemver.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(z,N,n), 
	DATA_TYPE POLYBENCH_1D(w,N,n), DATA_TYPE POLYBENCH_1D(v1,N,n),	DATA_TYPE POLYBENCH_1D(v2,N,n), DATA_TYPE POLYBENCH_1D(u1,N,n), DATA_TYPE POLYBENCH_1D(u2,N,n))
{
 	int i, j;

  	for (i = 0; i < N; i++)
	{
		u1[i] = i;
    		u2[i] = (i+1)/N/2.0;
    		v1[i] = (i+1)/N/4.0;
    		v2[i] = (i+1)/N/6.0;
    		y[i] = (i+1)/N/8.0;
    		z[i] = (i+1)/N/9.0;
    		x[i] = 0.0;
    		w[i] = 0.0;

    		for (j = 0; j < N; j++)
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

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("device id is %d\n",device_id);

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(z,N,n), 
	DATA_TYPE POLYBENCH_1D(w,N,n), DATA_TYPE POLYBENCH_1D(v1,N,n), DATA_TYPE POLYBENCH_1D(v2,N,n), DATA_TYPE POLYBENCH_1D(u1,N,n), DATA_TYPE POLYBENCH_1D(u2,N,n))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	z_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	w_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	u1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	u2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, B, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, z_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, z, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, w, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u2, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "gemver_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "gemver_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	// Create the OpenCL kernel
	clKernel3 = clCreateKernel(clProgram, "gemver_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	int n = N;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	
	/* Start timer. */
  	polybench_start_instruments;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&v1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&v2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&u1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&u2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	size_t global_item_size = sizeof(DATA_TYPE) * N; 
	size_t local_item_size = 64; 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clEnqueueBarrier(clCommandQue);

	int dim = N;
	
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&y_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(cl_mem), (void *)&z_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 4, sizeof(DATA_TYPE), (void *)&BETA);
	errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	clEnqueueBarrier(clCommandQue);
	
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&w_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 3, sizeof(DATA_TYPE), (void *)&ALPHA);
	errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
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
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(x_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void gemver(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(z,N,n), 
	DATA_TYPE POLYBENCH_1D(w,N,n), DATA_TYPE POLYBENCH_1D(v1,N,n), DATA_TYPE POLYBENCH_1D(v2,N,n), DATA_TYPE POLYBENCH_1D(u1,N,n), DATA_TYPE POLYBENCH_1D(u2,N,n))
{
	int i,j;
	
  	for (i = 0; i < N; i++)
	{
    		for (j = 0; j < N; j++)
		{
      			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

  	for (i = 0; i < N; i++)
	{
    		for (j = 0; j < N; j++)
		{
      			x[i] = x[i] + BETA * A[j][i] * y[j];
		}
	}

  	for (i = 0; i < N; i++)
	{
    		x[i] = x[i] + z[i];
	}

  	for (i = 0; i < N; i++)
	{
    		for (j = 0; j < N; j++)
		{
      			w[i] = w[i] +  ALPHA * A[i][j] * x[j];
		}
	}
}

void cl_launch_kernel1(
	int start_cpu,
	int end_cpu,
	int gpu_rows,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n), 
	DATA_TYPE POLYBENCH_1D(v1,N,n), 
	DATA_TYPE POLYBENCH_1D(v2,N,n), 
	DATA_TYPE POLYBENCH_1D(u1,N,n), 
	DATA_TYPE POLYBENCH_1D(u2,N,n)
) {
	if (gpu_rows > 0) {

		int n = N;
		size_t localWorkSize[2], globalWorkSize[2];
		localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
		localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
		globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
		globalWorkSize[1] = (size_t)ceil(((float)gpu_rows) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;

		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&v1_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&v2_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&u1_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&u2_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		size_t global_item_size = sizeof(DATA_TYPE) * N; 
		size_t local_item_size = 64; 
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	}

	int i,j;
	
  	for (i = start_cpu; i < end_cpu; i++)
	{
    		for (j = 0; j < N; j++)
		{
      			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

	if (gpu_rows > 0) {
		clFinish(clCommandQue);

		errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * gpu_rows * N, A, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

		errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in writing GPU mem\n");
	}

}

void cl_launch_kernel2(
	int start_cpu,
	int end_cpu,
	int gpu_rows,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
	DATA_TYPE POLYBENCH_1D(x,N,n), 
	DATA_TYPE POLYBENCH_1D(y,N,n),
	DATA_TYPE POLYBENCH_1D(z,N,n)
) {
	if (gpu_rows > 0) {
		int n = N;
		size_t localWorkSize[2], globalWorkSize[2];

		localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
		localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
		globalWorkSize[0] = (size_t)ceil(((float)gpu_rows) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
		globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;

		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&x_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&y_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 3, sizeof(cl_mem), (void *)&z_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 4, sizeof(DATA_TYPE), (void *)&BETA);
		errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	}
	int i,j;
	for (i = start_cpu; i < end_cpu; i++)
	{
    		for (j = 0; j < N; j++)
		{
      			x[i] = x[i] + BETA * A[j][i] * y[j];
		}
	}

  	for (i = start_cpu; i < end_cpu; i++)
	{
    		x[i] = x[i] + z[i];
	}

	if (gpu_rows > 0) {
		clFinish(clCommandQue);

		errcode = clEnqueueReadBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * gpu_rows, x, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

		errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in writing GPU mem\n");
	}
}

void cl_launch_kernel3(
	int start_cpu,
	int end_cpu,
	int gpu_rows,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n), 
	DATA_TYPE POLYBENCH_1D(x,N,n),
	DATA_TYPE POLYBENCH_1D(w,N,n)
) {
	if (gpu_rows > 0) {
		int n = N;
		size_t localWorkSize[2], globalWorkSize[2];

		localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
		localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
		globalWorkSize[0] = (size_t)ceil(((float)gpu_rows) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
		globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&x_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&w_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 3, sizeof(DATA_TYPE), (void *)&ALPHA);
		errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&n);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
	}

	int i,j;
	for (i = start_cpu; i < end_cpu; i++)
	{
		for (j = 0; j < N; j++)
		{
				w[i] = w[i] +  ALPHA * A[i][j] * x[j];
		}
	}

	if (gpu_rows > 0) {
		clFinish(clCommandQue);

		errcode = clEnqueueReadBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * gpu_rows, w, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
		
	}
}

void gemver_collab(
	int start_cpu,
	int end_cpu,
	int gpu_rows,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n), 
	DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
	DATA_TYPE POLYBENCH_1D(x,N,n), 
	DATA_TYPE POLYBENCH_1D(y,N,n), 
	DATA_TYPE POLYBENCH_1D(z,N,n), 
	DATA_TYPE POLYBENCH_1D(w,N,n), 
	DATA_TYPE POLYBENCH_1D(v1,N,n), 
	DATA_TYPE POLYBENCH_1D(v2,N,n), 
	DATA_TYPE POLYBENCH_1D(u1,N,n), 
	DATA_TYPE POLYBENCH_1D(u2,N,n)
) {
	cl_launch_kernel1(start_cpu, end_cpu, gpu_rows, A, v1, v2, u1, u2);
	cl_launch_kernel2(start_cpu, end_cpu, gpu_rows, A, x, y, z);
	cl_launch_kernel3(start_cpu, end_cpu, gpu_rows, A, x, w);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, w[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[]) 
{

	float a = 0.5f;// Input alpha parameter for splitting the work

	if (argc > 1) {
		a = atof(argv[1]);
	}

	// Calculate work distribution between CPU and GPU
	int cpu_start = (int)((1.0f - a) * N);
	int cpu_end = N;
	int cpu_rows = cpu_end - cpu_start;

	int gpu_start = 0;
	int gpu_end = cpu_start;
	int gpu_rows = gpu_end - gpu_start;

	printf("cpu_start = %d, cpu_end = %d, cpu_rows = %d\n", cpu_start, cpu_end, cpu_rows);
	printf("gpu_start = %d, gpu_end = %d, gpu_rows = %d\n", gpu_start, gpu_end, gpu_rows);
	printf("\n");

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
  	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
  	POLYBENCH_1D_ARRAY_DECL(w,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(z,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(u1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(u2,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v2,DATA_TYPE,N,n);
	
	init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z), POLYBENCH_ARRAY(w), 
		POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(u2));
	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z), POLYBENCH_ARRAY(w), 
		POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(u2));
	cl_load_prog();

	/* Start timer. */
  	polybench_start_instruments;

	gemver_collab(cpu_start, cpu_end, gpu_rows, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z), 
		POLYBENCH_ARRAY(w_outputFromGpu), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(u2));

	/* Stop and print timer. */
	printf("\nCPU-GPU Time in seconds: ");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	size_t size_b1 = 2 * sizeof(DATA_TYPE) * N * N;
	size_t size_b2 = 8 * sizeof(DATA_TYPE);

	size_t buffer_size = size_b1 + size_b2;
	size_t arg_size = sizeof(DATA_TYPE)*2 + sizeof(int);

	size_t total_bytes = buffer_size + arg_size;
	printf("Total bytes: %ld\n", total_bytes);

	size_t wg_size = DIM_LOCAL_WORK_GROUP_KERNEL_1_X * DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	printf("Work group size: %ld\n", wg_size);

	#ifdef RUN_ON_CPU

	    init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z), POLYBENCH_ARRAY(w), 
		POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(u2));
	
		/* Start timer. */
	  	polybench_start_instruments;

		gemver(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z), POLYBENCH_ARRAY(w), 
			POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(u2));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
  		polybench_stop_instruments;
 		polybench_print_instruments;

		compareResults(POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(N, POLYBENCH_ARRAY(w_outputFromGpu));

	#endif //RUN_ON_CPU


	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(w);  
	POLYBENCH_FREE_ARRAY(w_outputFromGpu);  
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(z);
	POLYBENCH_FREE_ARRAY(u1);
	POLYBENCH_FREE_ARRAY(u2);
	POLYBENCH_FREE_ARRAY(v1);
	POLYBENCH_FREE_ARRAY(v2);
	
    	return 0;
}

#include "../../common/polybench.c"
