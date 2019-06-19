
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include"device_launch_parameters.h"
#include"cuda_runtime.h"
#include<device_functions.h>
#include <stdint.h>

typedef uint8_t BYTE;
typedef uint32_t DWORD;
typedef int32_t LONG;
typedef int64_t LONGLONG;

typedef union _LARGE_INTEGER {
  struct {
    DWORD LowPart;
    LONG  HighPart;
  };
  struct {
    DWORD LowPart;
    LONG  HighPart;
  } u;
  LONGLONG QuadPart;
} LARGE_INTEGER, *PLARGE_INTEGER;
 
#define MASK_WIDTH 5
int filter_size = MASK_WIDTH;
int arr_size = 4096;
int res_size = arr_size;
#define O_TILE_WIDTH 64
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
 
 
void Conv2(float** filter, float** arr, float** res, int filter_size, int arr_size) {
	int temp;
 
	for (int i = 0; i<arr_size; i++) {
		for (int j = 0; j<arr_size; j++) {
			temp = 0;
			int starti = i - filter_size / 2;
			int startj = j - filter_size / 2;
			for (int m = starti; m<starti + filter_size; m++) {
				for (int n = startj; n<startj + filter_size; n++) {
					if (m >= 0 && m<arr_size&&n >= 0 && n<arr_size) {
						temp += filter[m - starti][n - startj] * arr[m][n];
					}
				}
			}
			res[i][j] = temp;
		}
	}
}
 
//kernel function
__global__
void convolution_2D_basic(float *in, float *out, float *mask, int maskwidth, int w, int h) {
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	if (Row < h&&Col < w) {
		float pixVal = 0;
		//start
		int startCol = Col - maskwidth / 2;
		int startRow = Row - maskwidth / 2;
		//caculate the res
		for (int i = 0; i < maskwidth; i++)
		{
			for (int j = 0; j < maskwidth; j++)
			{
				int curRow = startRow + i;
				int curCol = startCol + j;
				if (curRow > -1 && curRow<h&&curCol>-1 && curCol < w)
				{
					pixVal += mask[i*maskwidth + j] * in[curRow*w + curCol];
				}
			}
		}
		out[Row*w + Col] = pixVal;
	}
}
 
 
//kernel function
__global__
void convolution_2D_shared(float *in, float *out, float *mask, int maskwidth, int w, int h) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;
	int row_i = row_o - maskwidth / 2;
	int col_i = col_o - maskwidth / 2;
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	if ((row_i >= 0) && (row_i < h) &&
		(col_i >= 0) && (col_i < w)) {
		Ns[ty][tx] = in[row_i * w + col_i];
	}
	else {
		Ns[ty][tx] = 0.0f;
	}
	float output = 0.0f;
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0; i < maskwidth; i++) {
			for (int j = 0; j < maskwidth; j++) {
				output += mask[i*maskwidth+j] * Ns[i + ty][j + tx];
			}
		}
	if (row_o < h && col_o < w)
	{
			out[row_o*w + col_o] = output;
	}
	}
}
 
int check(float *a, float *b, int arr_size_1D)
{
	float res = 0;
	for (int i = 0; i<arr_size_1D; i++)
	{
		res += (a[i] - b[i]);
	}
	if ((res - 0)<1e-7)
		return 1;
	return 0;
}
__global__ void test()
{
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	printf("%d,%d]\n", Row, Col);
	printf("%d,%d,%d)\n", blockDim.y, blockDim.x, blockDim.z);
	printf("%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
}
float  pFilter[4096][4096],arr[4096][4096],res[4096][4096];
int  main()
{
	printf("the mask(filter) size is :%d X %d.\n", filter_size, filter_size);
	printf("the matrix size is :%d X %d.\n", arr_size, arr_size);
	clock_t start_CPU, end_CPU;
 
	// //


	
	//arr res pFilter
	int arr_size_1D = arr_size*arr_size;
	int filter_size_1D = filter_size*filter_size;
	float *arr_1D = (float*)malloc(arr_size_1D * sizeof(float));
	float *arr_1D_Cpu = (float*)malloc(arr_size_1D * sizeof(float));
	float *res_1D = (float*)malloc(arr_size_1D * sizeof(float));
	float *filter1D = (float*)malloc(filter_size_1D * sizeof(float));
 
 
	//allocate mem
	float *inD, *outD, *maskD;
	LARGE_INTEGER  num;
	long long start, end, freq;
	
	freq = num.QuadPart;
	start = num.QuadPart;
 
	//malloc
	cudaMalloc((void**)&inD, sizeof(float)*arr_size_1D);
	cudaMalloc((void**)&outD, sizeof(float)*arr_size_1D);
	cudaMalloc((void**)&maskD, sizeof(float*)*filter_size_1D);
 
	//copy
	cudaMemcpy(inD, arr_1D, sizeof(float)*arr_size_1D, cudaMemcpyHostToDevice);
	cudaMemcpy(outD, arr_1D, sizeof(float)*arr_size_1D, cudaMemcpyHostToDevice);
	cudaMemcpy(maskD, filter1D, sizeof(float)*filter_size_1D, cudaMemcpyHostToDevice);
	//kerner function void convolution_2D_basic(float *in,float *out,float *mask,int maskwidth,int w,int h)
	
	// int threadPerBlockX = 16;
	// int threadPerBlockY = 16;
	// dim3 grid((arr_size - 1) / threadPerBlockX + 1,(arr_size - 1) / threadPerBlockY + 1,1);
	// dim3 block(threadPerBlockX, threadPerBlockY);
	// convolution_2D_basic << <grid, block >>>(inD, outD, maskD, filter_size, arr_size, arr_size);
 
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((arr_size - 1) / O_TILE_WIDTH + 1, (arr_size - 1) / O_TILE_WIDTH + 1, 1);
    start_CPU = clock();
	convolution_2D_shared << <dimGrid, dimBlock >> >(inD, outD, maskD, filter_size, arr_size, arr_size);
 
	//copy back
	cudaMemcpy(res_1D, outD, sizeof(float)*arr_size_1D, cudaMemcpyDeviceToHost);
	printf("-------------------GPU version Done!------------------\n");
	end_CPU = clock();
	float time2= (float)(end_CPU - start_CPU) / CLOCKS_PER_SEC;
	printf("GPU time:%f ms\n", time2*1000);
	
	
	cudaFree(inD);
	cudaFree(outD);
	cudaFree(maskD);
 
 
	//check the res;
	//check(arr_1D,res_1D,arr_size_1D);
	printf("the check result is : %d\n", check(res_1D, arr_1D_Cpu, arr_size_1D));
//	printf("the speed up ratio is :%.2f\n", time*1000/ ((end_CPU - start_CPU) * 1000 * 1.0 / CLOCKS_PER_SEC));
	for (int i = 0; i<arr_size_1D; i++)
	{
		//printf("%.2f ", res_1D[i]);
	}
	
}