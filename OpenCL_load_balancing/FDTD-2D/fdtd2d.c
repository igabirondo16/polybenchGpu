/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "fdtd2d.h"
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

// DATA_TYPE alpha = 23; // Coefficient, not workload split alpha
DATA_TYPE beta_coeff = 15; // Renamed to avoid confusion if needed, though beta is not used in kernels
                           // The coefficients 0.5 and 0.7 are hardcoded in kernels.

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

cl_mem fict_mem_obj;
cl_mem ex_mem_obj;
cl_mem ey_mem_obj;
cl_mem hz_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU


void compareResults(DATA_TYPE POLYBENCH_2D(hz1,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz2,NX,NY,nx,ny))
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < NX; i++)
	{
		for (j=0; j < NY; j++)
		{
			if (percentDiff(hz1[i][j], hz2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
	fp = fopen("fdtd2d.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int i, j;

  	for (i = 0; i < TMAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}

	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i][j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i][j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i][j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
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


void cl_mem_init(DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	fict_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * TMAX, NULL, &errcode);
	ex_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	ey_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	hz_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, fict_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * TMAX, _fict_, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, ex_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, ex, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, ey_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, ey, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, hz, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "fdtd_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "fdtd_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel3 = clCreateKernel(clProgram, "fdtd_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue);
}

// GPU launcher for fdtd_kernel1 (updates ey)
void launch_gpu_fdtd_kernel1(int t, int gpu_rows_nx, int total_nx, int total_ny)
{
    if (gpu_rows_nx <= 0) return;
    size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};
    // Global work size should cover the number of GPU rows for the first dimension (NX-based)
    // and full columns for the second dimension (NY-based)
    size_t globalWorkSize[2] = {
        (size_t)ceil(((float)total_ny) / ((float)localWorkSize[0])) * localWorkSize[0], // Extent for NY
        (size_t)ceil(((float)gpu_rows_nx) / ((float)localWorkSize[1])) * localWorkSize[1]  // Extent for GPU's NX portion
    };

    errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&fict_mem_obj);
    errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&ex_mem_obj);
    errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&ey_mem_obj);
    errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&hz_mem_obj);
    errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&t);
    errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&total_nx); // Full NX dimension
    errcode |= clSetKernelArg(clKernel1, 6, sizeof(int), (void *)&total_ny); // Full NY dimension
    // Note: Kernels need to be written to respect gpu_rows_nx internally if not processing full NX.
    // However, PolyBench kernels usually take full dimensions and global_id determines behavior.
    // The current fdtd2d.cl kernels seem to use get_global_id(1) for row (i) and get_global_id(0) for col (j).
    // So, globalWorkSize[1] for rows should be gpu_rows_nx.
    // And kernel needs to know this boundary or be offset if not processing from row 0.
    // For simplicity, PolyBench kernels are often written assuming global_id(1) < actual_rows_to_process.
    // So gpu_rows_nx is passed to kernel or used to limit globalWorkSize[1].
    // The provided fdtd2d.cl takes total_nx and total_ny. The work is restricted by global size.
    // The 5th arg to clKernel1 in original code was gpu_rows_nx, 6th was total_ny.
    // Let's stick to convention: kernel knows total problem size, GPU work items are limited for its part.
    // If kernel is `kernel(..., int actual_rows_this_wg_processes, int total_cols)`
    // If kernel is `kernel(..., int total_rows_problem, int total_cols_problem)` and relies on `get_global_id`
    // The latter is more common in PolyBench. Arg 5 for kernel1 was NX, Arg 6 was NY in original context.
    // My previous launchers had gpu_rows_nx as arg 5. This is crucial.
    // Reverting to pass gpu_rows_nx to kernel if it expects it to limit its own loops.
    // Looking at fdtd2d.cl:
    // kernel1: ey[i*ny+j], ex[i*ny+j]. i = get_global_id(1), j = get_global_id(0).
    // It does not take gpu_rows_nx. It assumes i < nx. So globalWorkSize[1] must be gpu_rows_nx.
    // This is what my globalWorkSize calculation does. Arguments 5 & 6 should be total NX, NY.
    if(errcode != CL_SUCCESS) { printf("Error in setting arguments for Kernel 1 (ey)\n"); return; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error in launching Kernel 1 (ey)\n"); return; }
}

// GPU launcher for fdtd_kernel2 (updates ex)
void launch_gpu_fdtd_kernel2(int gpu_rows_nx, int total_nx, int total_ny)
{
    if (gpu_rows_nx <= 0) return;
    size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};
    size_t globalWorkSize[2] = {
        (size_t)ceil(((float)total_ny) / ((float)localWorkSize[0])) * localWorkSize[0], // NY extent
        (size_t)ceil(((float)gpu_rows_nx) / ((float)localWorkSize[1])) * localWorkSize[1]  // NX extent for GPU
    };
    // Kernel 2 in fdtd2d.cl: hz[i*ny+j]. i = get_global_id(1), j = get_global_id(0).
    // Needs total_nx, total_ny as arguments.
    errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
    errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
    errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
    errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&total_nx); // Full NX
    errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&total_ny); // Full NY
    if(errcode != CL_SUCCESS) { printf("Error in setting arguments for Kernel 2 (hz)\n"); return; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error in launching Kernel 2 (hz)\n"); return; }
}

// GPU launcher for fdtd_kernel3 (updates hz)
void launch_gpu_fdtd_kernel3(int gpu_rows_nx, int total_nx, int total_ny)
{
    if (gpu_rows_nx <= 0) return;
    size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};
    size_t globalWorkSize[2] = {
        (size_t)ceil(((float)total_ny) / ((float)localWorkSize[0])) * localWorkSize[0], // NY extent
        (size_t)ceil(((float)gpu_rows_nx) / ((float)localWorkSize[1])) * localWorkSize[1]  // NX extent for GPU
    };
    // Kernel 3 in fdtd2d.cl: hz[i*ny+j]. i = get_global_id(1), j = get_global_id(0).
    // Needs total_nx, total_ny as arguments.
    errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
    errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
    errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
    errcode |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&total_nx); // Full NX
    errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&total_ny); // Full NY
    if(errcode != CL_SUCCESS) { printf("Error in setting arguments for Kernel 3 (hz)\n"); return; }

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errcode != CL_SUCCESS) { printf("Error in launching Kernel 3 (hz)\n"); return; }
}

// Kernel 3 (source injection hz[0][0] = _fict[t]) is handled by CPU after hz sync,
// or by a very specific small GPU kernel if preferred. The generic launch below is not for that.
// For now, we assume kernel3 from the original code is not used in this CPU/GPU split logic,
// as its described functionality (if it's the main HZ update) is covered by kernel2.
// If fdtd2d.cl has three distinct computation kernels, this needs revisiting.
// Based on typical PolyBench FDTD-2D, there are two main field update steps.

void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(fict_mem_obj);
	errcode = clReleaseMemObject(ex_mem_obj);
	errcode = clReleaseMemObject(ey_mem_obj);
	errcode = clReleaseMemObject(hz_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

// CPU helper function for updating EY field (maps to part of fdtd_kernel1 in OpenCL)
// Updates ey for rows from cpu_row_start to cpu_row_start + num_cpu_rows - 1
void fdtd_cpu_ey_partial(DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax), DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                         DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny), int t, int total_nx, int total_ny,
                         int cpu_row_start, int num_cpu_rows)
{
    int i_cpu, j;
    for (i_cpu = 0; i_cpu < num_cpu_rows; i_cpu++)
    {
        int i_global = cpu_row_start + i_cpu;
        // Boundary conditions and updates for ey
        for (j = 0; j < total_ny; j++)
		{
			if (i_global == 0) // Boundary condition for ey at i=0
			{
				ey[i_global][j] = _fict_[t];
			}
			else if (i_global < total_nx) // Standard update for ey (i from 1 to NX-1)
			{
				ey[i_global][j] = ey[i_global][j] - 0.5f * (hz[i_global][j] - hz[i_global-1][j]);
			}
		}
    }
}

// CPU helper function for updating EX field (maps to part of fdtd_kernel1 in OpenCL)
// Updates ex for rows from cpu_row_start to cpu_row_start + num_cpu_rows - 1
void fdtd_cpu_ex_partial(DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                         int total_nx, int total_ny, int cpu_row_start, int num_cpu_rows)
{
    int i_cpu, j;
    for (i_cpu = 0; i_cpu < num_cpu_rows; i_cpu++)
    {
        int i_global = cpu_row_start + i_cpu;
        if (i_global < total_nx) // Standard update for ex (i from 0 to NX-1)
        {
            for (j = 1; j < total_ny; j++) // j from 1 to NY-1 for ex
			{
				ex[i_global][j] = ex[i_global][j] - 0.5f * (hz[i_global][j] - hz[i_global][j-1]);
			}
        }
    }
}

// CPU helper function for updating HZ field (maps to fdtd_kernel2 in OpenCL)
// Updates hz for rows from cpu_row_start to cpu_row_start + num_cpu_rows - 1
void fdtd_cpu_hz_partial(DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                         DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), int total_nx, int total_ny,
                         int cpu_row_start, int num_cpu_rows)
{
    int i_cpu, j;
    for (i_cpu = 0; i_cpu < num_cpu_rows; i_cpu++)
    {
        int i_global = cpu_row_start + i_cpu;
        if (i_global < total_nx -1) // Standard update for hz (i from 0 to NX-2)
        {
            for (j = 0; j < total_ny - 1; j++) // j from 0 to NY-2 for hz
			{
				hz[i_global][j] = hz[i_global][j] - 0.7f * (ex[i_global][j+1] - ex[i_global][j] + ey[i_global+1][j] - ey[i_global][j]);
			}
        }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
         fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

void runFdtd(DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int t, i, j;
	
	for(t=0; t < TMAX; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0][j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i][j] = ey[i][j] - 0.5*(hz[i][j] - hz[(i-1)][j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i][j] = ex[i][j] - 0.5*(hz[i][j] - hz[i][(j-1)]);
			}
		}

		for (i = 0; i < NX-1; i++)
		{
			for (j = 0; j < NY-1; j++)
			{
				hz[i][j] = hz[i][j] - 0.7*(ex[i][(j+1)] - ex[i][j] + ey[(i+1)][j] - ey[i][j]);
			}
		}
	}
}

int main(int argc, char *argv[])
{
	double alpha_cpu_split = 0.5; // Default alpha for workload splitting
	if (argc > 1) {
		alpha_cpu_split = atof(argv[1]);
		if (alpha_cpu_split < 0.0 || alpha_cpu_split > 1.0) {
			fprintf(stderr, "Alpha (for splitting) must be between 0.0 and 1.0\n");
			return 1;
		}
	}

	double alpha_gpu_split = 1.0f - alpha_cpu_split;

	// Calculate work distribution between CPU and GPU for NX dimension
	int gpu_nx_rows = (int)(NX * alpha_gpu_split);
	int cpu_nx_rows = NX - gpu_nx_rows;
	int cpu_row_start = gpu_nx_rows; // CPU starts after GPU rows

	printf("Workload Split Alpha: %f\n", alpha_cpu_split);
	printf("GPU rows (NX): 0 to %d (%d rows)\n", gpu_nx_rows -1, gpu_nx_rows);
	printf("CPU rows (NX): %d to %d (%d rows)\n", cpu_row_start, NX - 1, cpu_nx_rows);
	printf("\n");

	POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);
	POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny);

	init_arrays(POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));
	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));
	cl_load_prog();

	/* Start timer. */
	polybench_start_instruments;

	int t;
	for(t = 0; t < TMAX; t++)
	{
		// Host arrays ex, ey, hz are assumed to be consistent with device versions at start of loop.

		// Step 1: Update EY and EX fields using HZ from previous timestep
		// CPU part for EY and EX
		if (cpu_nx_rows > 0) {
			fdtd_cpu_ey_partial(POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), t, NX, NY, cpu_row_start, cpu_nx_rows);
			fdtd_cpu_ex_partial(POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(hz), NX, NY, cpu_row_start, cpu_nx_rows);
		}
		// GPU part for EY and EX (Kernel 1 in OpenCL updates both)
		if (gpu_nx_rows > 0) {
			launch_gpu_fdtd_kernel1(t, gpu_nx_rows, NX, NY);
			launch_gpu_fdtd_kernel2(gpu_nx_rows, NX, NY);
		}
		clFinish(clCommandQue); // Ensure GPU kernel is done

		// Consolidate EY and EX:
		// Read GPU-computed portions of EY and EX to host
		if (gpu_nx_rows > 0) {
			// Read ey[0...gpu_nx_rows-1] from GPU
			errcode = clEnqueueReadBuffer(clCommandQue, ey_mem_obj, CL_TRUE, 0, gpu_nx_rows * NY * sizeof(DATA_TYPE), POLYBENCH_ARRAY(ey), 0, NULL, NULL);
			if(errcode != CL_SUCCESS) { printf("Error reading ey_mem_obj (GPU part)\n"); return 1;}
			// Read ex[0...gpu_nx_rows-1] from GPU
			errcode = clEnqueueReadBuffer(clCommandQue, ex_mem_obj, CL_TRUE, 0, gpu_nx_rows * NY * sizeof(DATA_TYPE), POLYBENCH_ARRAY(ex), 0, NULL, NULL);
			if(errcode != CL_SUCCESS) { printf("Error reading ex_mem_obj (GPU part)\n"); return 1;}
		}
		clFinish(clCommandQue); // Ensure reads are complete. Host 'ex' and 'ey' are now fully updated.


		// Write updated EX and EY from host to device for HZ computation.
        // GPU already has its part correct. CPU part needs to be written if CPU did work.
        if (cpu_nx_rows > 0 && gpu_nx_rows > 0) {
			DATA_TYPE *ex_src_ptr = (DATA_TYPE*)POLYBENCH_ARRAY(ex) + (cpu_row_start * NY);
            errcode = clEnqueueWriteBuffer(clCommandQue, ex_mem_obj, CL_TRUE, cpu_row_start * NY * sizeof(DATA_TYPE),
                                           cpu_nx_rows * NY * sizeof(DATA_TYPE), ex_src_ptr, 0, NULL, NULL);
            if(errcode != CL_SUCCESS) { printf("Error writing CPU slice of ex to device\n"); return 1;}


			DATA_TYPE *ey_src_ptr = (DATA_TYPE*)POLYBENCH_ARRAY(ey) + (cpu_row_start * NY);
            errcode = clEnqueueWriteBuffer(clCommandQue, ey_mem_obj, CL_TRUE, cpu_row_start * NY * sizeof(DATA_TYPE),
                                           cpu_nx_rows * NY * sizeof(DATA_TYPE), ey_src_ptr, 0, NULL, NULL);
            if(errcode != CL_SUCCESS) { printf("Error writing CPU slice of ey to device\n"); return 1;}
        }
		clFinish(clCommandQue); // Ensure writes of CPU parts (if any) are complete. Device ex/ey are now fully consistent.

    
		// Step 2: Update HZ field using updated EX and EY
		// CPU part for HZ
		if (cpu_nx_rows > 0) {
            // Adjust num_cpu_rows for hz if it would go out of bounds (hz is NX-1 in i)
            //int actual_cpu_hz_rows = (cpu_row_start + cpu_nx_rows > NX -1 && cpu_row_start < NX -1) ? (NX - 1 - cpu_row_start) : cpu_nx_rows;
            // if (cpu_row_start >= NX -1) actual_cpu_hz_rows = 0; // No rows for CPU if start is too high
//
			//if (actual_cpu_hz_rows > 0) {
			fdtd_cpu_hz_partial(POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), NX, NY, cpu_row_start, cpu_nx_rows);
			//}
		}
		// GPU part for HZ (Kernel 2 in OpenCL)
		if (gpu_nx_rows > 0) {
            // Adjust gpu_nx_rows for HZ kernel if it would cause reading ey[gpu_nx_rows] when gpu_nx_rows == NX
            // The OpenCL kernel for HZ iterates i from 0 to NX-2. So global size for rows should be at most NX-1.
            // If gpu_nx_rows is NX, the effective number of rows processed by hz kernel is NX-1.
            //int gpu_hz_proc_rows = (gpu_nx_rows == NX) ? NX-1 : gpu_nx_rows;
            //if (gpu_hz_proc_rows > 0) { // only launch if there are rows to process for hz
			launch_gpu_fdtd_kernel3(gpu_nx_rows, NX, NY);
            //}
		}
		clFinish(clCommandQue); // Ensure GPU kernel is done.

		// Consolidate HZ:
		// Read GPU-computed portion of HZ to host
		if (gpu_nx_rows > 0) {
            //int gpu_hz_proc_rows = (gpu_nx_rows == NX) ? NX-1 : gpu_nx_rows;
            //if (gpu_hz_proc_rows > 0) {
			// Read hz[0...gpu_hz_proc_rows-1] from GPU
			errcode = clEnqueueReadBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, gpu_nx_rows * NY * sizeof(DATA_TYPE), POLYBENCH_ARRAY(hz), 0, NULL, NULL);
			if(errcode != CL_SUCCESS) { printf("Error reading hz_mem_obj (GPU part)\n"); return 1;}
            //}
		}
		clFinish(clCommandQue); // Ensure read is complete. Host 'hz' is now fully updated.

        // Source injection: hz[0][0] = _fict_[t]; (Standard PolyBench does not have this here, but in init or as boundary)
        // The OpenCL kernel1 handles ey[0][j] = _fict_[t]. HZ source is not typical for Polybench fdtd-2d kernel.
        // If there's a specific source injection for hz[0][0] like in some other versions:
        // POLYBENCH_ARRAY_2D(hz,NX,NY,nx,ny)[0][0] = POLYBENCH_ARRAY_1D(_fict_,TMAX,tmax)[t];

		// Write updated HZ from host to device for the next timestep.
        // GPU already has its part correct. CPU part needs to be written if CPU did work.
        if (cpu_nx_rows > 0) {
             // If CPU computed part of HZ, write that part to device.
             // Note: actual_cpu_hz_rows might be less than cpu_nx_rows due to boundary.
             // We need to be careful about the exact slice of hz that CPU modified.
             // fdtd_cpu_hz_partial modifies from cpu_row_start up to cpu_row_start + actual_cpu_hz_rows -1.
            //int actual_cpu_hz_rows = (cpu_row_start + cpu_nx_rows > NX -1 && cpu_row_start < NX -1) ? (NX - 1 - cpu_row_start) : cpu_nx_rows;
            //if (cpu_row_start >= NX -1) actual_cpu_hz_rows = 0;
//
            //if (actual_cpu_hz_rows > 0) {
			DATA_TYPE *hz_src_ptr = (DATA_TYPE*)POLYBENCH_ARRAY(hz) + (cpu_row_start * NY);
			errcode = clEnqueueWriteBuffer(clCommandQue, hz_mem_obj, CL_TRUE, cpu_row_start * NY * sizeof(DATA_TYPE),
											cpu_nx_rows * NY * sizeof(DATA_TYPE), hz_src_ptr, 0, NULL, NULL);
			if(errcode != CL_SUCCESS) { printf("Error writing CPU slice of hz to device\n"); return 1;}
            //}

			clFinish(clCommandQue);
        }
		// // Ensure write of CPU part (if any) is complete. Device hz is now fully consistent.
		
	}

	/* Stop and print timer. */
	printf("\nCPU-GPU Time in seconds: ");
	polybench_stop_instruments;
	polybench_print_instruments;

	size_t fict_size = sizeof(DATA_TYPE) * TMAX;
	size_t other_buffers = 3 * sizeof(DATA_TYPE) * NX * NY;
	size_t buffer_size = fict_size + other_buffers;
	size_t arg_size = 3*sizeof(int);

	size_t total_bytes = buffer_size + arg_size;
	printf("Total bytes: %ld\n", total_bytes);

	size_t wg_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
	printf("Work group size: %ld\n", wg_size);

	// Read the final hz array from host memory (it's already synced) for comparison or printing
	// POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny); is already declared
    // No, hz_outputFromGpu is for the original pure GPU version. We use host `hz` now.

	#ifdef RUN_ON_CPU
		// Create a separate hz_cpu array for the reference CPU computation
		POLYBENCH_2D_ARRAY_DECL(ex_cpu,DATA_TYPE,NX,NY,nx,ny);
		POLYBENCH_2D_ARRAY_DECL(ey_cpu,DATA_TYPE,NX,NY,nx,ny);
		POLYBENCH_2D_ARRAY_DECL(hz_cpu,DATA_TYPE,NX,NY,nx,ny);
		// Re-initialize arrays for CPU run
		init_arrays(POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex_cpu), POLYBENCH_ARRAY(ey_cpu), POLYBENCH_ARRAY(hz_cpu));

		/* Start timer. */
	  	polybench_start_instruments;

		// Perform full computation on CPU for reference
		// Original runFdtd needs to be defined or use a full CPU version of kernel loops
		// For now, assuming runFdtd is the original full CPU version.
		// Need to make sure 'runFdtd' is the original sequential CPU version.
		// Let's rename the original runFdtd to runFdtd_original_cpu for clarity if it's not already the case
		// For this exercise, assume runFdtd is the correct full CPU version.
		runFdtd(POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex_cpu), POLYBENCH_ARRAY(ey_cpu), POLYBENCH_ARRAY(hz_cpu));


		/* Stop and print timer. */
		printf("CPU Time in seconds: ");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(POLYBENCH_ARRAY(hz_cpu), POLYBENCH_ARRAY(hz)); // Compare reference CPU hz with combined hz

		POLYBENCH_FREE_ARRAY(ex_cpu);
		POLYBENCH_FREE_ARRAY(ey_cpu);
		POLYBENCH_FREE_ARRAY(hz_cpu);

	#else //print output to stderr so no dead code elimination

		print_array(NX, NY, POLYBENCH_ARRAY(hz)); // Print the combined hz

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(_fict_);
	POLYBENCH_FREE_ARRAY(ex);
	POLYBENCH_FREE_ARRAY(ey);
	POLYBENCH_FREE_ARRAY(hz);
	POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

	cl_clean_up();

    	return 0;
}

#include "../../common/polybench.c"

