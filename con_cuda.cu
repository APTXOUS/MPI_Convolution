#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <pthread.h>
#include <vector>
#include <unistd.h>
#include <time.h>
#pragma pack(1)
#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#define GRIDDIM_X 64
#define GRIDDIM_Y 64
#define MASK_WIDTH 5

__constant__ int d_const_Gaussian[MASK_WIDTH * MASK_WIDTH]; //分配常数存储器

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

static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char *d_src_imgbuf, unsigned char *d_dst_imgbuf)
{
	const int tix = blockDim.x * blockIdx.x + threadIdx.x;
	const int tiy = blockDim.y * blockIdx.y + threadIdx.y;

	const int threadTotalX = blockDim.x * gridDim.x;
	const int threadTotalY = blockDim.y * gridDim.y;

	for (int ix = tix; ix < height; ix += threadTotalX)
		for (int iy = tiy; iy < width; iy += threadTotalY)
		{
			for (int k = 0; k < byteCount; k++)
			{
				int sum = 0; //临时值
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//边界处理，幽灵元素赋值为零
						if (ix + m < 0 || iy + n < 0 || ix + m >= height || iy + n >= width)
							tempPixelValue = 0;
						else
							tempPixelValue = *(d_src_imgbuf + (ix + m) * width * byteCount + (iy + n) * byteCount + k);
						sum += tempPixelValue * d_const_Gaussian[(m + 2) * 5 + n + 2];
					}
				}

				if (sum / 273 < 0)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 0;
				else if (sum / 273 > 255)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 255;
				else
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = sum / 273;
			}
		}
}

static __global__ void max_pooling(int width, int height, int byteCount, unsigned char *d_src_imgbuf, unsigned char *d_dst_imgbuf)
{
	const int tix = blockDim.x * blockIdx.x + threadIdx.x;
	const int tiy = blockDim.y * blockIdx.y + threadIdx.y;

	const int threadTotalX = blockDim.x * gridDim.x;
	const int threadTotalY = blockDim.y * gridDim.y;

	for (int ix = tix; ix < height; ix += threadTotalX)
		for (int iy = tiy; iy < width; iy += threadTotalY)
		{
			for (int k = 0; k < byteCount; k++)
			{
				unsigned char temp = 0; //临时值
				unsigned char max_char = 0;
				for (int m = 0; m <= 1; m++)
				{
					for (int n = 0; n <= 1; n++)
					{
						temp = *(d_src_imgbuf + (ix * 2 + m) * width * 2 * byteCount + (iy * 2 + n) * byteCount + k);
						if (temp > max_char)
							max_char = temp;
					}
				}

				*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = max_char;
			}
		}
}



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

int main()
{
	//查看显卡配置
	struct cudaDeviceProp pror;
	cudaGetDeviceProperties(&pror, 0);

	unsigned char *h_src_imgbuf; //图像指针
	int width, height, byteCount;
	char rootPath1[] = "./img/timg.bmp";

	//h_src_imgbuf = readBmp(rootPath1, &width, &height, &byteCount);


	BITMAPFILEHEADER BmpHead;
	BITMAPINFOHEADER BmpInfo;

	FILE *fp = fopen(rootPath1, "rb"); //二进制读方式打开指定的图像文件
	if (fp == 0)
	{
			cerr << "Can not open " << rootPath1 << endl;
			return 0;
	}
	//获取位图文件头结构BITMAPFILEHEADER
	fread(&BmpHead, sizeof(BITMAPFILEHEADER), 1, fp);

	//获取图像宽、高、每像素所占位数等信息
	fread(&BmpInfo, sizeof(BITMAPINFOHEADER), 1, fp);
	width = BmpInfo.biWidth;   //宽度用来计算每行像素的字节数
	height = BmpInfo.biHeight; // 像素的行数
	byteCount= BmpInfo.biBitCount;

	readBmp(fp, h_src_imgbuf, width, height, byteCount, 0, height - 1);

	byteCount= BmpInfo.biBitCount/8;

	int size1 = width * height * byteCount * sizeof(unsigned char);
	int size2 = width * height * byteCount * sizeof(unsigned char) / 4; //max pooling 2*2

	printf("the matrix size is :%d X %d.\n", width, height);


	//输出图像内存-host端
	unsigned char *h_guassian_imgbuf = new unsigned char[width * height * byteCount];
	unsigned char *h_guassian_imgbuf_pooling = new unsigned char[width * height * byteCount / 4];

	//分配显存空间
	unsigned char *d_src_imgbuf;
	unsigned char *d_guassian_imgbuf;
	unsigned char *d_guassian_imgbuf_pooling;

	cudaMalloc((void **)&d_src_imgbuf, size1);
	cudaMalloc((void **)&d_guassian_imgbuf, size1);
	cudaMalloc((void **)&d_guassian_imgbuf_pooling, size2);
	

	//把数据从Host传到Device
	cudaMemcpy(d_src_imgbuf, h_src_imgbuf, size1, cudaMemcpyHostToDevice);


	//将高斯模板传入constant memory
	int Gaussian[25] = {1, 4, 7, 4, 1,
						4, 16, 26, 16, 4,
						7, 26, 41, 26, 7,
						4, 16, 26, 16, 4,
						1, 4, 7, 4, 1}; //总和为273
	cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

	int bx = ceil((double)width / BLOCKDIM_X); //网格和块的分配
	int by = ceil((double)height / BLOCKDIM_Y);

	if (bx > GRIDDIM_X)
		bx = GRIDDIM_X;
	if (by > GRIDDIM_Y)
		by = GRIDDIM_Y;

	dim3 grid(bx, by);					//网格的结构
	dim3 block(BLOCKDIM_X, BLOCKDIM_Y); //块的结构

	//CUDA计时函数
	cudaEvent_t start, stop; //CUDA计时机制
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//kernel--高斯滤波
	kernel_GaussianFilt<<<grid, block>>>(width, height, byteCount, d_src_imgbuf, d_guassian_imgbuf);
	cudaMemcpy(h_guassian_imgbuf, d_guassian_imgbuf, size1, cudaMemcpyDeviceToHost); //数据传回主机端

	// //max-pooling
	bx = ceil((double)width / 2 / BLOCKDIM_X); //网格和块的分配
	by = ceil((double)height / 2 / BLOCKDIM_Y);

	if (bx > GRIDDIM_X)
		bx = GRIDDIM_X;
	if (by > GRIDDIM_Y)
		by = GRIDDIM_Y;

	int width2 = width / 2;
	int height2 = height / 2;

	max_pooling<<<grid, block>>>(width2, height2, byteCount, d_guassian_imgbuf, d_guassian_imgbuf_pooling);
	cudaMemcpy(h_guassian_imgbuf_pooling, d_guassian_imgbuf_pooling, size2, cudaMemcpyDeviceToHost); //数据传回主机端

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("卷积部分所用时间为：%f ms\n", elapsedTime);

	saveBmp("result_other_pool.bmp", h_guassian_imgbuf_pooling, width2, height2, byteCount*8);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//释放内存
	cudaFree(d_src_imgbuf);
	cudaFree(d_guassian_imgbuf);

	delete[] h_src_imgbuf;
	delete[] h_guassian_imgbuf;
}