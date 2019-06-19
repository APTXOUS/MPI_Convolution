#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include <stdint.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <unistd.h>
#pragma pack(1)

typedef struct BITMAPFILEHEADER
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;

typedef struct BITMAPINFOHEADER
{
    uint32_t biSize;
    uint32_t biWidth;
    uint32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    uint32_t biXPelsPerMeter;
    uint32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;

#define MASK_WIDTH 5
int filter_size = MASK_WIDTH;
int arr_height = 4096;
int arr_width;
int arr_size=4096;
int res_size = arr_height;
#define O_TILE_WIDTH 64
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define BMP_FILE_NAME "./img/timg.bmp"


using namespace std;
float GsCore[25]={
	0.01441881,0.02808402,0.03507270,0.02808402,0.01441881,
	0.02808402,0.0547002, 0.06831229,0.0547002 ,0.02808402, 
	0.0350727 ,0.06831229,0.08531173,0.06831229,0.03507270,
	0.02808402,0.0547002 ,0.06831229,0.0547002 ,0.02808402, 
	0.01441881,0.02808402,0.03507270,0.02808402,0.01441881 
};



void Conv2(float **filter, float **arr, float **res, int filter_size, int arr_size)
{
	int temp;

	for (int i = 0; i < arr_size; i++)
	{
		for (int j = 0; j < arr_size; j++)
		{
			temp = 0;
			int starti = i - filter_size / 2;
			int startj = j - filter_size / 2;
			for (int m = starti; m < starti + filter_size; m++)
			{
				for (int n = startj; n < startj + filter_size; n++)
				{
					if (m >= 0 && m < arr_size && n >= 0 && n < arr_size)
					{
						temp += filter[m - starti][n - startj] * arr[m][n];
					}
				}
			}
			res[i][j] = temp;
		}
	}
}

//kernel function
__global__ void convolution_2D_basic(float *in, float *out, float *mask, int maskwidth, int w, int h)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	if (Row < h && Col < w)
	{
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
				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
				{
					pixVal += mask[i * maskwidth + j] * in[curRow * w + curCol];
				}
			}
		}
		out[Row * w + Col] = pixVal;
	}
}

//kernel function
__global__ void convolution_2D_shared(float *in, float *out, float *mask, int maskwidth, int w, int h)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;
	int row_i = row_o - maskwidth / 2;
	int col_i = col_o - maskwidth / 2;
	out[row_o * w + col_o] = in[row_o * w + col_o];
	// __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	// if ((row_i >= 0) && (row_i < h) &&
	// 		(col_i >= 0) && (col_i < w))
	// {
	// 	Ns[ty][tx] = in[row_i * w + col_i];
	// }
	// else
	// {
	// 	Ns[ty][tx] = 0.0f;
	// }
	// float output = 0.0f;
	// if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
	// {
	// 	for (int i = 0; i < maskwidth; i++)
	// 	{
	// 		for (int j = 0; j < maskwidth; j++)
	// 		{
	// 			output += mask[i * maskwidth + j] * Ns[i + ty][j + tx];
	// 		}
	// 	}
	// 	if (row_o < h && col_o < w)
	// 	{
	// 		out[row_o * w + col_o] = output;
	// 	}
	// }
}

int check(float *a, float *b, int arr_size_1D)
{
	float res = 0;
	for (int i = 0; i < arr_size_1D; i++)
	{
		res += (a[i] - b[i]);
	}
	if ((res - 0) < 1e-7)
		return 1;
	return 0;
}
__global__ void test()
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	printf("%d,%d]\n", Row, Col);
	printf("%d,%d,%d)\n", blockDim.y, blockDim.x, blockDim.z);
	printf("%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
}

void readBmp(FILE *fp,  unsigned char *&pBmpBuf, int BmpWidth, int BmpHeight,int BiBitCount, int startx, int endx)
{
	/**
* 灰度图像有颜色表，且颜色表表项为256
* (可以理解为lineByte是对bmpWidth的以4为步长的向上取整)
*/
	int lineByte = (BmpWidth * BiBitCount / 8 + 3) / 4 * 4;

	//申请位图数据所需要的空间，读位图数据进内存
	pBmpBuf = new (nothrow) unsigned char[lineByte * BmpHeight];

	if (pBmpBuf == NULL)
	{
		cerr << "Mem alloc failed." << endl;
		exit(-1);
	}
	if (startx - 2 > 0)
		startx = startx - 2;
	if (endx + 2 < BmpHeight)
		endx = endx + 2;

	fseek(fp, startx * lineByte, SEEK_CUR);
	cerr<<fread(pBmpBuf + startx * lineByte, lineByte * (endx - startx + 1), 1, fp)<<endl;

	return;
}


//给定一个图像位图数据、宽、高、颜色表指针及每像素所占的位数等信息,将其写到指定文件中
bool saveBmp(const char *bmpName, unsigned char *imgBuf, int width, int height,
	int biBitCount)
{
//如果位图数据指针为0，则没有数据传入，函数返回
if (!imgBuf)
return 0;

//颜色表大小，以字节为单位，灰度图像颜色表为1024字节，彩色图像颜色表大小为0
int colorTablesize = 0;

if (biBitCount == 8)
colorTablesize = 1024; // 8*128

//待存储图像数据每行字节数为4的倍数
int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;

//以二进制写的方式打开文件
FILE *fp = fopen(bmpName, "wb");

if (fp == 0)
{
cerr << "Open file error." << endl;
return 0;
}

//申请位图文件头结构变量，填写文件头信息
BITMAPFILEHEADER fileHead;

fileHead.bfType = 0x4D42; // bmp类型

// bfSize是图像文件4个组成部分之和
fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) +
					 colorTablesize + lineByte * height;
fileHead.bfReserved1 = 0;
fileHead.bfReserved2 = 0;

// bfOffBits是图像文件前3个部分所需空间之和
fileHead.bfOffBits = 54 + colorTablesize;

//写文件头进文件
fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

//申请位图信息头结构变量，填写信息头信息
BITMAPINFOHEADER head;

head.biBitCount = biBitCount;
head.biClrImportant = 0;
head.biClrUsed = 0;
head.biCompression = 0;
head.biHeight = height;
head.biPlanes = 1;
head.biSize = 40;
head.biSizeImage = lineByte * height;
head.biWidth = width;
head.biXPelsPerMeter = 0;
head.biYPelsPerMeter = 0;

//写位图信息头进内存
fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

//写位图数据进文件
fwrite(imgBuf, height * lineByte, 1, fp);

//关闭文件
fclose(fp);

return 1;
}

int main()
{
	unsigned char *bmpBuf;
	unsigned char *result;
	
	int BmpWidth;   //图像的宽
	int BmpHeight;  //图像的高
	int BiBitCount; //图像类型，每像素位数 8-灰度图 24-彩色图
	BITMAPFILEHEADER BmpHead;
	BITMAPINFOHEADER BmpInfo;

	FILE *fp = fopen(BMP_FILE_NAME, "rb"); //二进制读方式打开指定的图像文件
	if (fp == 0)
	{
			cerr << "Can not open " << BMP_FILE_NAME << endl;
			return 0;
	}
	//获取位图文件头结构BITMAPFILEHEADER
	cerr <<fread(&BmpHead, sizeof(BITMAPFILEHEADER), 1, fp);

	//获取图像宽、高、每像素所占位数等信息
	cerr <<fread(&BmpInfo, sizeof(BITMAPINFOHEADER), 1, fp);

	BmpWidth = BmpInfo.biWidth;   //宽度用来计算每行像素的字节数
	BmpHeight = BmpInfo.biHeight; // 像素的行数

	//计算图像每行像素所占的字节数（必须是4的倍数）
	BiBitCount = BmpInfo.biBitCount;

	int lineByte = (BmpWidth * BiBitCount / 8 + 3) / 4 * 4;

	// 将图片读取到内存中
	result = new(nothrow) unsigned char[BmpHeight * lineByte];

	readBmp(fp, bmpBuf, BmpWidth, BmpHeight, BiBitCount, 0, BmpHeight - 1);

	printf("the mask(filter) size is :%d X %d.\n", filter_size, filter_size);
	printf("the matrix size is :%d X %d.\n", BmpWidth, BmpHeight);


	clock_t start_CPU, end_CPU;

	//arr res pFilter
	int arr_size_1D = BmpWidth * BmpHeight;
	int filter_size_1D = filter_size * filter_size;
	float *arr_1Dr = (float *)malloc(arr_size_1D * sizeof(float));
	float *arr_1Dg = (float *)malloc(arr_size_1D * sizeof(float));
	float *arr_1Db = (float *)malloc(arr_size_1D * sizeof(float));

	float *res_1Dr = (float *)malloc(arr_size_1D * sizeof(float));
	float *res_1Dg = (float *)malloc(arr_size_1D * sizeof(float));
	float *res_1Db = (float *)malloc(arr_size_1D * sizeof(float));
	

	int i=0;
	while(i<arr_size_1D)
	{
		arr_1Dr[i]=bmpBuf[i*3];
		arr_1Dg[i]=bmpBuf[i*3+1];
		arr_1Db[i]=bmpBuf[i*3+2];
		i++;
	}

	// ############################初始化完毕############################

	
	// ############################以下是卷积部分############################

	//allocate mem
	float *inD, *outD, *maskD;
	//malloc 分配cuda内存
	cudaMalloc((void **)&inD, sizeof(float) * arr_size_1D);
	cudaMalloc((void **)&outD, sizeof(float) * arr_size_1D);
	cudaMalloc((void **)&maskD, sizeof(float *) * filter_size_1D);


	//复制图片
	cudaMemcpy(inD, arr_1Dr, sizeof(float) * arr_size_1D, cudaMemcpyHostToDevice);
	//cudaMemcpy(outD, arr_1Dr, sizeof(float) * arr_size_1D, cudaMemcpyHostToDevice);
	//复制卷积核
	cudaMemcpy(maskD, GsCore, sizeof(float) * filter_size_1D, cudaMemcpyHostToDevice);
	//kerner function void convolution_2D_basic(float *in,float *out,float *mask,int maskwidth,int w,int h)

	// int threadPerBlockX = 16;
	// int threadPerBlockY = 16;
	// dim3 grid((arr_size - 1) / threadPerBlockX + 1,(arr_size - 1) / threadPerBlockY + 1,1);
	// dim3 block(threadPerBlockX, threadPerBlockY);
	// convolution_2D_basic << <grid, block >>>(inD, outD, maskD, filter_size, arr_size, arr_size);

	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid((arr_size - 1) / O_TILE_WIDTH + 1, (arr_size - 1) / O_TILE_WIDTH + 1, 1);
	start_CPU = clock();
	convolution_2D_shared<<<dimGrid, dimBlock>>>(inD, outD, maskD, filter_size, BmpWidth, BmpHeight);
	//copy back

	cudaMemcpy(res_1Dr, outD, sizeof(float) * arr_size_1D, cudaMemcpyDeviceToHost);

	// cudaMemcpy(inD, arr_1Dg, sizeof(float) * arr_size_1D, cudaMemcpyHostToDevice);
	// convolution_2D_shared<<<dimGrid, dimBlock>>>(inD, outD, maskD, filter_size, BmpWidth, BmpHeight);
	// //copy back
	// cudaMemcpy(res_1Dg, outD, sizeof(float) * arr_size_1D, cudaMemcpyDeviceToHost);

	// cudaMemcpy(inD, arr_1Db, sizeof(float) * arr_size_1D, cudaMemcpyHostToDevice);
	// convolution_2D_shared<<<dimGrid, dimBlock>>>(inD, outD, maskD, filter_size, BmpWidth, BmpHeight);
	// //copy back
	// cudaMemcpy(res_1Db, outD, sizeof(float) * arr_size_1D, cudaMemcpyDeviceToHost);

	i=0;
	while(i<arr_size_1D)
	{
		result[i*3]=res_1Dr[i];
		result[i*3+1]=res_1Dg[i];
		result[i*3+2]=res_1Db[i];
		i=i+1;
	}


	printf("-------------------GPU version Done!------------------\n");
	end_CPU = clock();
	float time2 = (float)(end_CPU - start_CPU) / CLOCKS_PER_SEC;
	printf("GPU time:%f ms\n", time2 * 1000);
	saveBmp("cudaresult.bmp", result, BmpWidth, BmpHeight, BiBitCount);
	cudaFree(inD);
	cudaFree(outD);
	cudaFree(maskD);

	//check the res;
	//check(arr_1D,res_1D,arr_size_1D);
	//printf("the check result is : %d\n", check(res_1D, arr_1D_Cpu, arr_size_1D));
	//	printf("the speed up ratio is :%.2f\n", time*1000/ ((end_CPU - start_CPU) * 1000 * 1.0 / CLOCKS_PER_SEC));
	for (int i = 0; i < arr_size_1D; i++)
	{
	//	printf("%.2f ", arr_1Dr[i]);
	}
}