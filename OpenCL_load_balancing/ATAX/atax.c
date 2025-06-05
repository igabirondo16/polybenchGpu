/**
 * atax.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <ctype.h> // For isdigit

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_CPU

#define RUN_ON_CPU // Ensure comparison is active

#include "atax.h" // Must be after NX, NY defines if they are in polybench.h via atax.h
#include "../../common/polybench.h" // For POLYBENCH_1D_ARRAY_DECL etc.
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#ifndef M_PI
#define M_PI 3.14159
#endif

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

// OpenCL global variables
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
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem tmp_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

// Global alpha for CPU share, to be set by main from argv
float alpha_cpu_share_global = 0.5f;

// Function prototypes
static void init_array(int nx, int ny, DATA_TYPE x_arr[NY], DATA_TYPE A_arr[NX][NY]);
static void atax_cpu_ref(int nx_ref, int ny_ref, DATA_TYPE A_ref_arr[NX][NY],
                         DATA_TYPE x_ref_arr[NY], DATA_TYPE y_ref_arr[NY],
		                 DATA_TYPE tmp_ref_arr[NX]);
static void atax_cpu_partial(int cpu_start_row, int cpu_end_row, int nx_total, int ny_total,
                             DATA_TYPE A_host_arr[NX][NY], DATA_TYPE x_host_arr[NY],
                             DATA_TYPE y_cpu_contrib_arr[NY], DATA_TYPE tmp_cpu_slice_arr[NX]);
static void print_array(int ny_print, DATA_TYPE y_to_print[NY]);


void compareResults(int ny_compare, DATA_TYPE y_ref[NY], DATA_TYPE y_collab[NY])
{
	int i, fail;
	fail = 0;
	for (i=0; i<ny_compare; i++)
	{
		if (percentDiff(y_ref[i], y_collab[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void read_cl_file()
{
	fp = fopen("atax.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel: atax.cl\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if (!source_str) { fprintf(stderr, "Failed to allocate memory for kernel source.\n"); exit(1); }
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
    if (source_size == 0) {
        fprintf(stderr, "Error: atax.cl is empty or could not be read properly.\n");
        exit(1);
    }
    if (source_size < MAX_SOURCE_SIZE) {
        source_str[source_size] = '\0';
    } else {
        source_str[MAX_SOURCE_SIZE - 1] = '\0';
    }
}

void cl_initialization()
{
    errcode = clGetPlatformIDs(0, NULL, &num_platforms);
    CL_CHECK(errcode);
    if (num_platforms == 0) {
        fprintf(stderr, "Error: Failed to find any OpenCL platforms.\n");
        exit(1);
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) { fprintf(stderr, "Failed to allocate memory for platforms.\n"); exit(1); }
    errcode = clGetPlatformIDs(num_platforms, platforms, NULL);
    CL_CHECK(errcode);

    platform_id = NULL;
    device_id = NULL;

    for (cl_uint i = 0; i < num_platforms; i++) {
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
                 free(devices_on_platform);
                 break;
            }
            free(devices_on_platform);
        }
    }
    free(platforms);

    if (device_id == NULL) {
        fprintf(stderr, "Error: No suitable OpenCL device found for type %lu.\n", (unsigned long)OPENCL_DEVICE_SELECTION);
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
}

void cl_mem_init(DATA_TYPE A_host[NX][NY], DATA_TYPE x_host[NY], DATA_TYPE y_on_host[NY], DATA_TYPE tmp_on_host[NX])
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DATA_TYPE) * NX * NY, A_host, &errcode); CL_CHECK(errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DATA_TYPE) * NY, x_host, &errcode); CL_CHECK(errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(DATA_TYPE) * NY, y_on_host, &errcode); CL_CHECK(errcode);
	tmp_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(DATA_TYPE) * NX, tmp_on_host, &errcode); CL_CHECK(errcode);
}

void cl_load_prog()
{
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
    CL_CHECK(errcode);

	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
    if (errcode != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *)malloc(log_size + 1);
        if(!build_log) { perror("Failed to allocate build_log"); exit(1); }
        clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        fprintf(stderr, "Error in building program: %s\n", build_log);
        free(build_log);
        CL_CHECK(errcode);
    }

	clKernel1 = clCreateKernel(clProgram, "atax_kernel1", &errcode); CL_CHECK(errcode);
	clKernel2 = clCreateKernel(clProgram, "atax_kernel2", &errcode); CL_CHECK(errcode);

    errcode = clFinish(clCommandQue); CL_CHECK(errcode);
}

void launch_atax_kernel1_gpu(int nx_orig, int ny_orig, int nx_gpu_rows)
{
	if (nx_gpu_rows <= 0) return;
	size_t localWorkSize[1] = {DIM_LOCAL_WORK_GROUP_X};
	size_t globalWorkSize[1] = {(size_t)ceil(((float)nx_gpu_rows) / localWorkSize[0]) * localWorkSize[0]};

	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&tmp_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&nx_orig); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&ny_orig); CL_CHECK(errcode);

	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CL_CHECK(errcode);
}

void launch_atax_kernel2_gpu(int nx_orig, int ny_orig, int ny_elements_for_y, int nx_rows_in_tmp)
{
	if (ny_elements_for_y <= 0) return;
	size_t localWorkSize[1] = {DIM_LOCAL_WORK_GROUP_X};
	size_t globalWorkSize[1] = {(size_t)ceil(((float)ny_elements_for_y) / localWorkSize[0]) * localWorkSize[0]};

	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&y_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&tmp_mem_obj); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&ny_orig); CL_CHECK(errcode);
	errcode = clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&nx_rows_in_tmp); CL_CHECK(errcode);

	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CL_CHECK(errcode);
}

void cl_clean_up()
{
	errcode = clFlush(clCommandQue); CL_CHECK(errcode);
	errcode = clFinish(clCommandQue); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel1); CL_CHECK(errcode);
	errcode = clReleaseKernel(clKernel2); CL_CHECK(errcode);
	errcode = clReleaseProgram(clProgram); CL_CHECK(errcode);
	errcode = clReleaseMemObject(a_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseMemObject(x_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseMemObject(y_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseMemObject(tmp_mem_obj); CL_CHECK(errcode);
	errcode = clReleaseCommandQueue(clCommandQue); CL_CHECK(errcode);
	errcode = clReleaseContext(clGPUContext); CL_CHECK(errcode);
}

// Static function definitions
static void init_array(int nx_param, int ny_param, DATA_TYPE x_arr[NY], DATA_TYPE A_arr[NX][NY])
{
	int i, j;
	for (i = 0; i < ny_param; i++)
	{
		x_arr[i] = (DATA_TYPE)i * M_PI;
	}
	for (i = 0; i < nx_param; i++)
	{
		for (j = 0; j < ny_param; j++)
		{
			A_arr[i][j] = ((DATA_TYPE) i*j) / NX;
		}
	}
}

static void atax_cpu_ref(int nx_ref, int ny_ref, DATA_TYPE A_ref_arr[NX][NY],
                         DATA_TYPE x_ref_arr[NY], DATA_TYPE y_ref_arr[NY],
		                 DATA_TYPE tmp_ref_arr[NX])
{
	int i,j;
	for (i= 0; i < ny_ref; i++)
	{
		y_ref_arr[i] = 0.0;
	}

	for (i = 0; i < nx_ref; i++)
 	{
		tmp_ref_arr[i] = 0.0;
		for (j = 0; j < ny_ref; j++)
		{
			tmp_ref_arr[i] = tmp_ref_arr[i] + A_ref_arr[i][j] * x_ref_arr[j];
		}
		for (j = 0; j < ny_ref; j++)
		{
			y_ref_arr[j] = y_ref_arr[j] + A_ref_arr[i][j] * tmp_ref_arr[i];
		}
    }
}

static void atax_cpu_partial(
  int cpu_start_row,
  int cpu_end_row,
  int nx_total,
  int ny_total,
  DATA_TYPE A_host_arr[NX][NY],
  DATA_TYPE x_host_arr[NY],
  DATA_TYPE y_cpu_contrib_arr[NY],
  DATA_TYPE tmp_cpu_slice_arr[NX]
)
{
  int i, j;
  for (i = cpu_start_row; i < cpu_end_row; i++) {
    tmp_cpu_slice_arr[i] = 0.0;
    for (j = 0; j < ny_total; j++) {
      tmp_cpu_slice_arr[i] += A_host_arr[i][j] * x_host_arr[j];
    }
  }

  for (j = 0; j < ny_total; j++) {
    for (i = cpu_start_row; i < cpu_end_row; i++) {
      y_cpu_contrib_arr[j] += A_host_arr[i][j] * tmp_cpu_slice_arr[i];
    }
  }
}

static
void print_array(int ny_print, DATA_TYPE y_to_print[NY])
{
  int i;
  for (i = 0; i < ny_print; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y_to_print[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}

int main(int argc, char *argv[])
{
	float alpha_val_local = 0.5f;
	if (argc > 1) {
        const char* arg_str = argv[1];
        int dot_count = 0;
        int valid_char = 1;
        char current_char_val;
        for (int i_arg = 0; (current_char_val = arg_str[i_arg]) != '\0'; i_arg++) {
            if (current_char_val == '.') {
                dot_count++;
            } else if (!isdigit(current_char_val) && !(i_arg == 0 && (current_char_val == '+' || current_char_val == '-'))) {
                 if (current_char_val != 'f' && current_char_val != 'F') {
                    valid_char = 0;
                    break;
                 }
            }
        }
        if (valid_char && dot_count <= 1) {
            alpha_val_local = atof(arg_str);
        } else {
            fprintf(stderr, "Warning: Invalid alpha input '%s'. Using default 0.5.\n", arg_str);
            alpha_val_local = 0.5f;
        }

		if (alpha_val_local < 0.0f || alpha_val_local > 1.0f) {
			fprintf(stderr, "ERROR: Alpha value (CPU share for NX) must be between 0.0 and 1.0. Using default 0.5.\n");
			alpha_val_local = 0.5f;
		}
	}
    alpha_cpu_share_global = alpha_val_local;

	int nx = NX;
	int ny = NY;

	int cpu_nx_rows = (int)(nx * alpha_cpu_share_global);
    int gpu_nx_rows = nx - cpu_nx_rows;
	int cpu_nx_row_start = gpu_nx_rows;

	printf("ATAX Benchmark with CPU-GPU Collaboration (CPU Share for NX rows: %.2f)\n", alpha_cpu_share_global);
	printf("NX (rows of A, tmp size): %d, NY (cols of A, y size): %d\n", nx, ny);
	printf("GPU processes first %d rows of A for kernel 1 (tmp calculation).\n", gpu_nx_rows);
	printf("CPU processes last %d rows of A (from row %d) for kernel 1 (tmp calculation).\n", cpu_nx_rows, cpu_nx_row_start);
	printf("\n");

	POLYBENCH_2D_ARRAY_DECL(A_host,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(x_host,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_ref_cpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_device_output,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_final_combined,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp_host_buffer,DATA_TYPE,NX,nx);

	for (int i = 0; i < ny; i++) {
		(*y_final_combined)[i] = 0.0f;
        (*y_ref_cpu)[i] = 0.0f;
	}
	for (int i = 0; i < nx; i++) {
		(*tmp_host_buffer)[i] = 0.0f;
	}

	init_array(nx, ny, POLYBENCH_ARRAY(x_host), POLYBENCH_ARRAY(A_host));

    if (gpu_nx_rows > 0) {
	    read_cl_file();
	    cl_initialization();
	    cl_mem_init(POLYBENCH_ARRAY(A_host), POLYBENCH_ARRAY(x_host), POLYBENCH_ARRAY(y_ref_cpu), POLYBENCH_ARRAY(tmp_host_buffer));
	    cl_load_prog();
    }

    polybench_start_instruments;

	if (cpu_nx_rows > 0) {
		atax_cpu_partial(cpu_nx_row_start, nx, nx, ny,
			POLYBENCH_ARRAY(A_host), POLYBENCH_ARRAY(x_host),
			POLYBENCH_ARRAY(y_final_combined), POLYBENCH_ARRAY(tmp_host_buffer)
		);
	}

	if (gpu_nx_rows > 0) {
		launch_atax_kernel1_gpu(nx, ny, gpu_nx_rows);
		errcode = clEnqueueBarrierWithWaitList(clCommandQue, 0, NULL, NULL); CL_CHECK(errcode);
		launch_atax_kernel2_gpu(nx, ny, ny, gpu_nx_rows);
		errcode = clFinish(clCommandQue); CL_CHECK(errcode);

		errcode = clEnqueueReadBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NY, (*y_device_output), 0, NULL, NULL);
        CL_CHECK(errcode);

		for (int i = 0; i < ny; i++) {
			(*y_final_combined)[i] += (*y_device_output)[i];
		}
	}

	polybench_stop_instruments;
	printf("Total (CPU+GPU) Time in seconds:\n");
    polybench_print_instruments;

	#ifdef RUN_ON_CPU
        POLYBENCH_1D_ARRAY_DECL(tmp_for_ref_cpu,DATA_TYPE,NX,nx);
        for(int i=0; i<nx; ++i) (*tmp_for_ref_cpu)[i] = 0.0;

	  	polybench_start_instruments;
		atax_cpu_ref(nx, ny, POLYBENCH_ARRAY(A_host), POLYBENCH_ARRAY(x_host), POLYBENCH_ARRAY(y_ref_cpu), POLYBENCH_ARRAY(tmp_for_ref_cpu));
		polybench_stop_instruments;
		printf("CPU Time in seconds:\n");
	 	polybench_print_instruments;

		compareResults(ny, POLYBENCH_ARRAY(y_ref_cpu), POLYBENCH_ARRAY(y_final_combined));
        POLYBENCH_FREE_ARRAY(tmp_for_ref_cpu);
	#else
		if (polybench_get_dump_array_setting()) {
            print_array(ny, POLYBENCH_ARRAY(y_final_combined));
        }
	#endif

    if (gpu_nx_rows > 0) {
	    cl_clean_up();
    }

	POLYBENCH_FREE_ARRAY(A_host);
	POLYBENCH_FREE_ARRAY(x_host);
	POLYBENCH_FREE_ARRAY(y_ref_cpu);
	POLYBENCH_FREE_ARRAY(y_device_output);
	POLYBENCH_FREE_ARRAY(y_final_combined);
	POLYBENCH_FREE_ARRAY(tmp_host_buffer);

	return 0;
}

#include "../../common/polybench.c"
