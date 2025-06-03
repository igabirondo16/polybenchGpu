/**
 * covariance.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <math.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_CPU

#include "covariance.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


char str_temp[1024];

DATA_TYPE float_n; // Will be initialized in main
DATA_TYPE eps=  0.005;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_reduce;
cl_kernel clKernel_covar;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("covariance.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
	int i, j;

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			data[i][j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	//if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	//else printf("Error getting platform IDs\n");
	if(errcode != CL_SUCCESS) printf("Error: clGetPlatformIDs returned %d\n", errcode);

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	//if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	//else printf("Error getting platform name\n");
	if(errcode != CL_SUCCESS) printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) returned %d\n", errcode);

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	//if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	//else printf("Error getting platform version\n");
	if(errcode != CL_SUCCESS) printf("Error: clGetPlatformInfo(CL_PLATFORM_VERSION) returned %d\n", errcode);

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	//if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	//else printf("Error getting device IDs\n");
	if(errcode != CL_SUCCESS) printf("Error: clGetDeviceIDs returned %d\n", errcode);

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	//if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	//else printf("Error getting device name\n");
	if(errcode != CL_SUCCESS) printf("Error: clGetDeviceInfo returned %d\n", errcode);
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	//if(errcode != CL_SUCCESS) printf("Error in creating context\n");
	if(errcode != CL_SUCCESS) printf("Error: clCreateContext returned %d\n", errcode);
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	//if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
	if(errcode != CL_SUCCESS) printf("Error: clCreateCommandQueue returned %d\n", errcode);
}


void cl_mem_init(DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m))
{
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateBuffer (data_mem_obj) returned %d\n", errcode);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * M, NULL, &errcode); // Corrected size M*M
	if(errcode != CL_SUCCESS) printf("Error: clCreateBuffer (symmat_mem_obj) returned %d\n", errcode);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateBuffer (mean_mem_obj) returned %d\n", errcode);
		
	//if(errcode != CL_SUCCESS) printf("Error in creating buffers\n"); // This check is redundant due to individual checks

	errcode = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, data, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error: clEnqueueWriteBuffer (data_mem_obj) returned %d\n", errcode);
	// errcode = clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * M, symmat, 0, NULL, NULL); // symmat is output, not written here initially
	// if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, mean, 0, NULL, NULL); // mean is output, not written here initially either, but kernel expects it. Let's keep this for now.
	if(errcode != CL_SUCCESS) printf("Error: clEnqueueWriteBuffer (mean_mem_obj) returned %d\n", errcode);
}

 
void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateProgramWithSource returned %d\n", errcode);

	// Build the program
	//errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
    errcode = clBuildProgram(clProgram, 1, &device_id, "-I ../../common", NULL, NULL); // Added include path for polybench.h
	if(errcode != CL_SUCCESS)
	{
		printf("Error: clBuildProgram returned %d\n", errcode);
		// Print build log
		size_t log_size;
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *)malloc(log_size);
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		free(log);
	}
		
	// Create the OpenCL kernel
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateKernel (mean_kernel) returned %d\n", errcode);

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateKernel (reduce_kernel) returned %d\n", errcode);

	clKernel_covar = clCreateKernel(clProgram, "covar_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error: clCreateKernel (covar_kernel) returned %d\n", errcode);

	clFinish(clCommandQue);	
}


void cl_launch_kernel1_mean(int cpu_features_start, int cpu_features_end, int gpu_features_count, int M_features, int N_datapoints,
                             DATA_TYPE POLYBENCH_2D(data_arr, N_datapoints, M_features, n_actual, m_actual),
                             DATA_TYPE POLYBENCH_1D(mean_arr, M_features, m_actual))
{
    size_t localWorkSize_Kernel1[2];
    size_t globalWorkSize_Kernel1[2];

    // GPU Part
    if (gpu_features_count > 0)
    {
        localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X; // Should be defined in covariance.h
        localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y; // Should be 1 for mean_kernel
        //globalWorkSize_Kernel1[0] = (size_t)ceil(((float)gpu_features_count) / DIM_LOCAL_WORK_GROUP_KERNEL_1_X) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
        //globalWorkSize_Kernel1[1] = 1; // Mean kernel is 1D in terms of features it processes independently

		// As per item 17
		globalWorkSize_Kernel1[0] = (size_t)ceil(((float)gpu_features_count) / localWorkSize_Kernel1[0]) * localWorkSize_Kernel1[0];
        globalWorkSize_Kernel1[1] = 1;


        errcode =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
        errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
        errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
        errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&M_features); // Total features
        errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&N_datapoints); // Total datapoints
		// Kernel needs to know which part to process if it's not processing all features.
		// Assuming mean_kernel calculates mean for features 0 to gpu_features_count-1
		// Or, the kernel is modified to take an offset, or data_mem_obj is offset.
		// For now, assume kernel calculates for first gpu_features_count features and stores them at start of mean_mem_obj
        if(errcode != CL_SUCCESS) printf("Error: clSetKernelArg (clKernel_mean) returned %d\n", errcode);

        errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueNDRangeKernel (clKernel_mean) returned %d\n", errcode);
    }

    // CPU Part
    // Calculate mean for features from cpu_features_start to cpu_features_end - 1
    for (int j = cpu_features_start; j < cpu_features_end; j++)
    {
        mean_arr[j] = 0.0;
        for (int i = 0; i < N_datapoints; i++)
        {
            mean_arr[j] += data_arr[i][j];
        }
        mean_arr[j] /= float_n;
    }

    // Sync and data transfer
    errcode = clFinish(clCommandQue);
    if(errcode != CL_SUCCESS) printf("Error: clFinish after mean kernel/CPU part returned %d\n", errcode);

    if (gpu_features_count > 0)
    {
        // Read the GPU computed part of mean_arr from mean_mem_obj
        // Assuming GPU computes first gpu_features_count elements of mean.
        errcode = clEnqueueReadBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * gpu_features_count, mean_arr, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueReadBuffer (mean_mem_obj to mean_arr) returned %d\n", errcode);
    }

    // Write the entire mean_arr (CPU part + GPU part) back to mean_mem_obj for the next kernel (reduce)
    // This assumes mean_arr now holds the complete, correct mean values.
    errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M_features, mean_arr, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) printf("Error: clEnqueueWriteBuffer (mean_arr to mean_mem_obj) returned %d\n", errcode);

    errcode = clFinish(clCommandQue); // Ensure write completes before next kernel uses it
    if(errcode != CL_SUCCESS) printf("Error: clFinish after writing combined mean_arr to buffer returned %d\n", errcode);
}


void cl_launch_kernel2_reduce(int cpu_datapoints_start, int cpu_datapoints_end, int gpu_datapoints_count, int M_features, int N_datapoints,
                               DATA_TYPE POLYBENCH_2D(data_arr, N_datapoints, M_features, n_actual, m_actual),
                               DATA_TYPE POLYBENCH_1D(mean_arr, M_features, m_actual)) // mean_arr is read-only here
{
    size_t localWorkSize_Kernel2[2];
    size_t globalWorkSize_Kernel2[2];

    // GPU Part
    if (gpu_datapoints_count > 0)
    {
        localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X; // From covariance.h
        localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y; // From covariance.h

        // As per item 17
        globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M_features) / localWorkSize_Kernel2[0]) * localWorkSize_Kernel2[0];
        globalWorkSize_Kernel2[1] = (size_t)ceil(((float)gpu_datapoints_count) / localWorkSize_Kernel2[1]) * localWorkSize_Kernel2[1];

        // mean_mem_obj should already contain the full mean vector from kernel1
        errcode =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
        errcode |= clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&data_mem_obj);
        errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(int), (void *)&M_features);
        errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(int), (void *)&N_datapoints);
        // Again, kernel might need modification to handle only a slice of datapoints (rows).
        // Assuming kernel processes the first gpu_datapoints_count rows of data_mem_obj.
        // Or an offset needs to be passed to the kernel or applied to data_mem_obj.
        if(errcode != CL_SUCCESS) printf("Error: clSetKernelArg (clKernel_reduce) returned %d\n", errcode);

        errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueNDRangeKernel (clKernel_reduce) returned %d\n", errcode);
    }

    // CPU Part
    // Reduce (subtract mean) for datapoints from cpu_datapoints_start to cpu_datapoints_end - 1
    for (int i = cpu_datapoints_start; i < cpu_datapoints_end; i++)
    {
        for (int j = 0; j < M_features; j++)
        {
            data_arr[i][j] -= mean_arr[j]; // mean_arr should be complete here
        }
    }

    // Sync and data transfer
    errcode = clFinish(clCommandQue);
    if(errcode != CL_SUCCESS) printf("Error: clFinish after reduce kernel/CPU part returned %d\n", errcode);

    if (gpu_datapoints_count > 0)
    {
        // Read the GPU processed part of data_arr from data_mem_obj
        // Assuming GPU processes first gpu_datapoints_count rows.
        // Data is read into the beginning of data_arr.
        errcode = clEnqueueReadBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0,
                                      sizeof(DATA_TYPE) * gpu_datapoints_count * M_features,
                                      data_arr, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueReadBuffer (data_mem_obj to data_arr) returned %d\n", errcode);
    }

    // Write the entire data_arr (CPU part + GPU part) back to data_mem_obj for the next kernel (covar)
    // This assumes data_arr now holds the complete, mean-subtracted data.
    errcode = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0,
                                   sizeof(DATA_TYPE) * N_datapoints * M_features,
                                   data_arr, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) printf("Error: clEnqueueWriteBuffer (data_arr to data_mem_obj) returned %d\n", errcode);

    errcode = clFinish(clCommandQue); // Ensure write completes
    if(errcode != CL_SUCCESS) printf("Error: clFinish after writing combined data_arr to buffer returned %d\n", errcode);
}

void cl_launch_kernel3_covar(int cpu_features_start, int cpu_features_end, int gpu_features_count, int M_features, int N_datapoints,
                              DATA_TYPE POLYBENCH_2D(data_arr, N_datapoints, M_features, n_actual, m_actual),      // Read-only, should be mean-centered
                              DATA_TYPE POLYBENCH_2D(symmat_arr, M_features, M_features, m_actual_sym, m_actual_sym_cols)) // Output
{
    size_t localWorkSize_Kernel3[2];
    size_t globalWorkSize_Kernel3[2];

    // GPU Part
    if (gpu_features_count > 0)
    {
        localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X; // From covariance.h
        localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y; // Should be 1 for covar_kernel (processes j1)

        // As per item 17
        globalWorkSize_Kernel3[0] = (size_t)ceil(((float)gpu_features_count) / localWorkSize_Kernel3[0]) * localWorkSize_Kernel3[0];
        globalWorkSize_Kernel3[1] = 1; // covar_kernel is 1D in terms of features (j1) it processes for symmat

        // data_mem_obj should contain the full mean-centered data from kernel2
        errcode =  clSetKernelArg(clKernel_covar, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
        errcode |= clSetKernelArg(clKernel_covar, 1, sizeof(cl_mem), (void *)&data_mem_obj);
        errcode |= clSetKernelArg(clKernel_covar, 2, sizeof(int), (void *)&M_features);
        errcode |= clSetKernelArg(clKernel_covar, 3, sizeof(int), (void *)&N_datapoints);
        // Kernel needs to know which part of symmat to output.
        // Assuming kernel computes rows 0 to gpu_features_count-1 of symmat.
        // Each such row symmat[j1] has M_features elements (symmat[j1][j2] for j2 from j1 to M_features-1, then mirrored).
        // The kernel as written in covariance.cl computes symmat[j1][j2] for j1 from 0..get_global_id(0)-1 and j2 from j1..M-1
        if(errcode != CL_SUCCESS) printf("Error: clSetKernelArg (clKernel_covar) returned %d\n", errcode);

        errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_covar, 1, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueNDRangeKernel (clKernel_covar) returned %d\n", errcode);
    }

    // CPU Part
    // Calculate covariance for features j1 from cpu_features_start to cpu_features_end - 1
    for (int j1 = cpu_features_start; j1 < cpu_features_end; j1++)
    {
        for (int j2 = j1; j2 < M_features; j2++)
        {
            symmat_arr[j1][j2] = 0.0;
            for (int i = 0; i < N_datapoints; i++)
            {
                symmat_arr[j1][j2] += data_arr[i][j1] * data_arr[i][j2]; // data_arr is mean-centered
            }
            // Ensuring strict float division if DATA_TYPE is float
            if (N_datapoints > 1) { // Match kernel's (n > 1) condition implicitly
                 symmat_arr[j1][j2] /= (float_n - 1.0f);
            }
            symmat_arr[j2][j1] = symmat_arr[j1][j2];
        }
    }

    // Sync and data transfer
    errcode = clFinish(clCommandQue);
    if(errcode != CL_SUCCESS) printf("Error: clFinish after covar kernel/CPU part returned %d\n", errcode);

    if (gpu_features_count > 0)
    {
        // Read the GPU computed part of symmat_arr from symmat_mem_obj
        // GPU computes first gpu_features_count rows (0 to gpu_features_count-1).
        // Each row has M_features columns.
        // The kernel computes the upper triangle part for these rows.
        // Reading gpu_features_count * M_features elements.
        errcode = clEnqueueReadBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0,
                                      sizeof(DATA_TYPE) * gpu_features_count * M_features,
                                      symmat_arr, 0, NULL, NULL);
        if(errcode != CL_SUCCESS) printf("Error: clEnqueueReadBuffer (symmat_mem_obj to symmat_arr) returned %d\n", errcode);
        // Note: The CPU part calculates its portion. The symmat_arr now has GPU computed rows [0..gpu_features_count-1]
        // and CPU computed rows [cpu_features_start..cpu_features_end-1].
        // If there's an overlap or gap, this logic is flawed.
        // Based on typical splits: gpu_features_count = alpha*M; cpu_features_start = gpu_features_count. No overlap.
        // The symmat_arr is now the final combined result.
    }
     // No need to write symmat_arr back to device as it's the final output for this sequence.
    errcode = clFinish(clCommandQue);
    if(errcode != CL_SUCCESS) printf("Error: clFinish after reading symmat_arr from buffer returned %d\n", errcode);
}


void covariance_collab(int gpu_M_features_count, int cpu_M_features_start, int cpu_M_features_end,
                       int gpu_N_datapoints_count, int cpu_N_datapoints_start, int cpu_N_datapoints_end,
                       int M_val, int N_val, // Changed M, N to M_val, N_val for clarity
                       DATA_TYPE POLYBENCH_2D(data_arr, N_val, M_val, pb_N, pb_M),
                       DATA_TYPE POLYBENCH_1D(mean_arr, M_val, pb_M),
                       DATA_TYPE POLYBENCH_2D(symmat_arr, M_val, M_val, pb_M, pb_M))
{
    cl_launch_kernel1_mean(cpu_M_features_start, cpu_M_features_end, gpu_M_features_count, M_val, N_val,
                           data_arr, mean_arr);

    cl_launch_kernel2_reduce(cpu_N_datapoints_start, cpu_N_datapoints_end, gpu_N_datapoints_count, M_val, N_val,
                             data_arr, mean_arr);

    cl_launch_kernel3_covar(cpu_M_features_start, cpu_M_features_end, gpu_M_features_count, M_val, N_val,
                            data_arr, symmat_arr);
}

/* Removed cl_launch_kernel */

void cl_clean_up()
{
	// Clean up
	cl_int status;
	status = clFlush(clCommandQue);
	if(status != CL_SUCCESS) printf("Error: clFlush returned %d\n", status);
	status = clFinish(clCommandQue);
	if(status != CL_SUCCESS) printf("Error: clFinish returned %d\n", status);
	status = clReleaseKernel(clKernel_reduce);
	if(status != CL_SUCCESS) printf("Error: clReleaseKernel (clKernel_reduce) returned %d\n", status);
	status = clReleaseKernel(clKernel_mean);
	if(status != CL_SUCCESS) printf("Error: clReleaseKernel (clKernel_mean) returned %d\n", status);
	status = clReleaseKernel(clKernel_covar);
	if(status != CL_SUCCESS) printf("Error: clReleaseKernel (clKernel_covar) returned %d\n", status);
	status = clReleaseProgram(clProgram);
	if(status != CL_SUCCESS) printf("Error: clReleaseProgram returned %d\n", status);
	status = clReleaseMemObject(symmat_mem_obj);
	if(status != CL_SUCCESS) printf("Error: clReleaseMemObject (symmat_mem_obj) returned %d\n", status);
	status = clReleaseMemObject(data_mem_obj);
	if(status != CL_SUCCESS) printf("Error: clReleaseMemObject (data_mem_obj) returned %d\n", status);
	status = clReleaseMemObject(mean_mem_obj);
	if(status != CL_SUCCESS) printf("Error: clReleaseMemObject (mean_mem_obj) returned %d\n", status);
	status = clReleaseCommandQueue(clCommandQue);
	if(status != CL_SUCCESS) printf("Error: clReleaseCommandQueue returned %d\n", status);
	status = clReleaseContext(clGPUContext);
	if(status != CL_SUCCESS) printf("Error: clReleaseContext returned %d\n", status);
}

// Modified covariance to use M_val, N_val parameters consistent with polybench macros _PB_M, _PB_N
// and to include (N-1) normalization, float_n is effectively N_val (number of data points)
void covariance(int M_val, int N_val, DATA_TYPE POLYBENCH_2D(data,N_val,M_val,m,n), DATA_TYPE POLYBENCH_2D(symmat,M_val,M_val,m,m), DATA_TYPE POLYBENCH_1D(mean,M_val,m))
{
	int i, j, j1,j2;
    DATA_TYPE current_float_n = (DATA_TYPE)N_val; // Use N_val for normalization factor

  	/* Determine mean of column vectors of input data matrix */
	for (j = 0; j < M_val; j++) // Iterates up to M_val (number of features)
	{
		mean[j] = 0.0;
		for (i = 0; i < N_val; i++) // Iterates up to N_val (number of data points)
		{
        		mean[j] += data[i][j];
		}
		mean[j] /= current_float_n;
	}

  	/* Center the column vectors. */
	for (i = 0; i < N_val; i++)
	{
		for (j = 0; j < M_val; j++)
		{
			data[i][j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 0; j1 < M_val; j1++)
	{
		for (j2 = j1; j2 < M_val; j2++) // Corrected loop bound for j2
     		{
       		symmat[j1][j2] = 0.0;
			for (i = 0; i < N_val; i++)
			{
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
            // Ensuring strict float division if DATA_TYPE is float
            if (N_val > 1) { // Match kernel's (n > 1) condition
                 symmat[j1][j2] /= (current_float_n - 1.0f);
            }
		symmat[j2][j1] = symmat[j1][j2];
      		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int M_val, DATA_TYPE POLYBENCH_2D(symmat,M_val,M_val,m,m_cols)) // Use M_val
{
  int i, j;

  for (i = 0; i < M_val; i++)
    for (j = 0; j < M_val; j++) { // Iterate M_val columns
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * M_val + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{	
	int M_val = M; // M from covariance.h, e.g., #define M 1024
	int N_val = N; // N from covariance.h, e.g., #define N 1024

    float_n = (DATA_TYPE)N_val; // Initialize float_n with N_val

    double alpha = 0.5; // Default alpha value for load balancing
    if (argc > 1) {
        alpha = atof(argv[1]);
    }
    if (alpha < 0.0 || alpha > 1.0) {
        fprintf(stderr, "INFO: Alpha value %.2f is out of [0.0, 1.0]. Using default 0.5.\n", alpha);
        alpha = 0.5;
    }

    // Calculate workload splits based on alpha
    int gpu_M_features_count = (int)(alpha * M_val);
    int cpu_M_features_start = gpu_M_features_count;
    int cpu_M_features_end = M_val;
    if (cpu_M_features_start > M_val) cpu_M_features_start = M_val; // clamp to M_val
    if (gpu_M_features_count < 0) gpu_M_features_count = 0; // clamp to 0
    if (gpu_M_features_count > M_val ) gpu_M_features_count = M_val;


    int gpu_N_datapoints_count = (int)(alpha * N_val);
    int cpu_N_datapoints_start = gpu_N_datapoints_count;
    int cpu_N_datapoints_end = N_val;
    if (cpu_N_datapoints_start > N_val) cpu_N_datapoints_start = N_val; // clamp to N_val
    if (gpu_N_datapoints_count < 0) gpu_N_datapoints_count = 0; // clamp to 0
    if (gpu_N_datapoints_count > N_val) gpu_N_datapoints_count = N_val;


    printf("Selected alpha: %.2f\n", alpha);
    printf("M_val (features): %d, N_val (datapoints): %d\n", M_val, N_val);
    printf("GPU M features: %d (indices 0 to %d)\n", gpu_M_features_count, (gpu_M_features_count > 0 ? gpu_M_features_count - 1 : -1));
    printf("CPU M features: %d (indices %d to %d)\n", (cpu_M_features_end - cpu_M_features_start), cpu_M_features_start, cpu_M_features_end - 1);
    printf("GPU N datapoints: %d (indices 0 to %d)\n", gpu_N_datapoints_count, (gpu_N_datapoints_count > 0 ? gpu_N_datapoints_count -1 : -1));
    printf("CPU N datapoints: %d (indices %d to %d)\n", (cpu_N_datapoints_end - cpu_N_datapoints_start), cpu_N_datapoints_start, cpu_N_datapoints_end - 1);

    // Adjust CPU start if GPU takes all work to prevent negative counts in printf
    if (gpu_M_features_count == M_val) cpu_M_features_start = M_val;
    if (gpu_N_datapoints_count == N_val) cpu_N_datapoints_start = N_val;


	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N_val,M_val,N_val,M_val);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M_val,M_val,M_val,M_val);
	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M_val,M_val);
	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M_val,M_val,M_val,M_val);

	init_arrays(N_val, M_val, POLYBENCH_ARRAY(data));
    for(int i=0; i<M_val; i++) (*mean)[i] = 0.0; // Initialize mean array
    // symmat_outputFromGpu will be entirely overwritten by collab function
    // symmat (for CPU ref) will be entirely overwritten by covariance function
    
	read_cl_file(); // source_str is allocated here
	cl_initialization();
    // cl_mem_init expects M, N order for polybench macros, but sizes are N*M for data, M*M for symmat, M for mean.
    // It uses M and N passed as arguments for sizes.
	cl_mem_init(POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat_outputFromGpu), POLYBENCH_ARRAY(mean));
	cl_load_prog();

    printf("\nStarting collaborative covariance calculation...\n");
	/* Start timer. */
	polybench_start_instruments;

    covariance_collab(gpu_M_features_count, cpu_M_features_start, cpu_M_features_end,
                      gpu_N_datapoints_count, cpu_N_datapoints_start, cpu_N_datapoints_end,
                      M_val, N_val,
                      POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(symmat_outputFromGpu));

	/* Stop and print timer. */
    printf("Collaborative GPU+CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	#ifdef RUN_ON_CPU
		printf("\nStarting CPU-only covariance calculation for comparison...\n");
		// Re-initialize data and mean for a fresh CPU run
		init_arrays(N_val, M_val, POLYBENCH_ARRAY(data));
        for(int i=0; i<M_val; i++) (*mean)[i] = 0.0;

		/* Start timer. */
	  	polybench_start_instruments;

		covariance(M_val, N_val, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

        // Pass M_val for both dimensions of symmat to compareResults
		compareResults(M_val, M_val, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	#else
        printf("\nRUN_ON_CPU not defined. Printing collaborative result (symmat_outputFromGpu) to stderr.\n");
		print_array(M_val, POLYBENCH_ARRAY(symmat_outputFromGpu));
	#endif


	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);	
    if (source_str != NULL) free(source_str); // Free memory allocated in read_cl_file
	
	return 0;
}

#include "../../common/polybench.c"

