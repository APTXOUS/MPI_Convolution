#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

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
} BITMAPINFODEADER;

using namespace std;

#define BMP_FILE_NAME "timg.bmp"

const int N = 5;
double GsCore[N][N];
unsigned char *pBmpBuf = NULL; //读入图像数据的指针
unsigned char *rBmpBuf = NULL;
unsigned char *gBmpBuf = NULL;
unsigned char *bBmpBuf = NULL;
/*******************************************************************************/

void showBmpHead(BITMAPFILEHEADER &pBmpHead)
{
    cout << "==========位图文件头==========" << endl;
    cout << "文件头类型:" << pBmpHead.bfType << endl;
    cout << "文件大小:" << pBmpHead.bfSize << endl;
    cout << "保留字_1:" << pBmpHead.bfReserved1 << endl;
    cout << "保留字_2:" << pBmpHead.bfReserved2 << endl;
    cout << "实际位图数据的偏移字节数:" << pBmpHead.bfOffBits << endl
         << endl;
}

void showBmpInforHead(BITMAPINFODEADER &pBmpInforHead)
{
    cout << "==========位图信息头==========" << endl;
    cout << "结构体的长度:" << pBmpInforHead.biSize << endl;
    cout << "位图宽:" << pBmpInforHead.biWidth << endl;
    cout << "位图高:" << pBmpInforHead.biHeight << endl;
    cout << "biPlanes平面数:" << pBmpInforHead.biPlanes << endl;
    cout << "biBitCount采用颜色位数:" << pBmpInforHead.biBitCount << endl;
    cout << "压缩方式:" << pBmpInforHead.biCompression << endl;
    cout << "biSizeImage实际位图数据占用的字节数:" << pBmpInforHead.biSizeImage << endl;
    cout << "X方向分辨率:" << pBmpInforHead.biXPelsPerMeter << endl;
    cout << "Y方向分辨率:" << pBmpInforHead.biYPelsPerMeter << endl;
    cout << "使用的颜色数:" << pBmpInforHead.biClrUsed << endl;
    cout << "重要颜色数:" << pBmpInforHead.biClrImportant << endl;
}

void readBmp(FILE *fp, unsigned char *&pBmpBuf, int BmpWidth, int BmpHeight,
             int BiBitCount)
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

    fread(pBmpBuf, lineByte * BmpHeight, 1, fp);

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
//将多通道颜色数据拆分成三个数组
void reFormChannel(int numWidth, int numHeigh)
{
    //新数组的高度和宽度
    int reFormHeigh = numHeigh + N / 2 + N / 2;
    int reFormWidth = numWidth + N / 2 + N / 2;

    //申请二维动态数组
    int i, j;
    rBmpBuf = new (nothrow) unsigned char[reFormHeigh * reFormWidth];
    gBmpBuf = new (nothrow) unsigned char[reFormHeigh * reFormWidth];
    bBmpBuf = new (nothrow) unsigned char[reFormHeigh * reFormWidth];

    //赋值并补0
    for (i = 0; i < reFormHeigh; i++)
    {
        for (j = 0; j < reFormWidth; j++)
        {
            if (i < N / 2 || i >= numHeigh + N / 2)
            {
                rBmpBuf[i * reFormWidth + j] = 0;
                gBmpBuf[i * reFormWidth + j] = 0;
                bBmpBuf[i * reFormWidth + j] = 0;
            }
            else if (j < N / 2 || j >= numWidth + N / 2)
            {
                rBmpBuf[i * reFormWidth + j] = 0;
                gBmpBuf[i * reFormWidth + j] = 0;
                bBmpBuf[i * reFormWidth + j] = 0;
            }
            else
            {
                rBmpBuf[i * reFormWidth + j] = pBmpBuf[(i - N / 2) * numWidth * 3 + (j - N / 2) * 3];
                gBmpBuf[i * reFormWidth + j] = pBmpBuf[(i - N / 2) * numWidth * 3 + (j - N / 2) * 3 + 1];
                bBmpBuf[i * reFormWidth + j] = pBmpBuf[(i - N / 2) * numWidth * 3 + (j - N / 2) * 3 + 2];
            }
        }
    }
}

void genGsCore()
{
    int i, j;
    double sigma = 1.5;
    double sum = 0.0;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            GsCore[i][j] =
                exp(-((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) /
                    (2.0 * sigma * sigma));
            sum += GsCore[i][j];
        }
    }
    FILE *fp;
    fp = fopen("gs.txt", "w");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            GsCore[i][j] /= sum;
            fprintf(fp, "%.8f ", GsCore[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int Value;
int len = N / 2;
int vvv = -len * Value;
unsigned char getValue(int ii, unsigned char *arrary)
{
    int h, k;
    double sum = 0;
    int vv = vvv;
    ii = ii - vvv ;//+ len;
    for (h = -len; h <= len; h++)
    {
        for (k = -len; k <= len; k++)
        {
            sum += arrary[vv + ii + k] * GsCore[-h + len][-k + len];
        }
        vv += Value;
    }
    return sum;
}

/**
 * 卷积公共计算部分
 */

unsigned char *convolution(unsigned char *resBuf,int start_x, int end_x, int BmpWidth)
{


    int ii;
    int reFormWidth = BmpWidth * 3;
    int endi = (end_x+1) * reFormWidth;
    
    int xx = start_x * (BmpWidth + N / 2 + N / 2)+N/2;
    int yy = 0;
    int xx_add = (BmpWidth + N / 2 + N / 2);
    int yy_v = BmpWidth;
    for (ii = start_x * reFormWidth; ii < endi; ii = ii + 3)
    {
        if (yy == yy_v)
        {
            xx = xx + xx_add;
            yy = 0;
        }
        //cout<<ii/3/4096<< " "<<ii/3%4096<<" "<<(xx-N/2)/4100<< " "<<yy<<endl;
        resBuf[ii] = getValue(xx + yy, rBmpBuf);
        resBuf[ii + 1] = getValue(xx + yy, gBmpBuf);
        resBuf[ii + 2] = getValue(xx + yy, bBmpBuf);
        yy++;
    }


    return 0;
}
double im2col_GsCore[N*N];
void im2col_core()
{
    int count=0;
    for(int i=N-1;i>=0;i--)
        for(int j=N-1;j>=0;j--)
        {
            im2col_GsCore[count]=GsCore[i][j];
            count++;
        }
}

unsigned char ** rCol;
unsigned char ** gCol;
unsigned char ** bCol;

void im2col(int numWidth, int numHeigh)
{
    rCol=new unsigned char *[numWidth*numHeigh];
    for(int i=0;i<numWidth*numHeigh;i++)
        rCol[i]=new unsigned char[N*N];
    gCol=new unsigned char *[numWidth*numHeigh];
    for(int i=0;i<numWidth*numHeigh;i++)
        gCol[i]=new unsigned char[N*N];
    bCol=new unsigned char *[numWidth*numHeigh];
    for(int i=0;i<numWidth*numHeigh;i++)
        bCol[i]=new unsigned char[N*N];
    int NN=N*N;
    int reFormWidth = numWidth +N-1;
    for(int i=0;i<numWidth*numHeigh;i++)
    {
        for(int j=-N/2;j<=N/2;j++)
            for(int k=-N/2;k<=N/2;k++)
            {
                rCol[i][(j+N/2)*N+(k+N/2)]=rBmpBuf[(i/numWidth+N/2+j)*reFormWidth+i%numWidth+N/2+k];
                gCol[i][(j+N/2)*N+(k+N/2)]=gBmpBuf[(i/numWidth+N/2+j)*reFormWidth+i%numWidth+N/2+k];
                bCol[i][(j+N/2)*N+(k+N/2)]=bBmpBuf[(i/numWidth+N/2+j)*reFormWidth+i%numWidth+N/2+k];
            }
    }


    
    return;
}
int NN=N*N;
int im2col_getValue(int num,unsigned char **arrary)
{
    double sum=0;
    for(int i=0;i<NN;i++)
        sum+=arrary[num][i]*im2col_GsCore[i];
    return sum;
}

int im2col_convolution(unsigned char *resBuf,int start_x, int end_x, int BmpWidth)
{


    int ii;
    int reFormWidth = BmpWidth * 3;
    int endi = (end_x+1) * reFormWidth;
    int num=start_x*BmpWidth;
    for (ii = start_x * reFormWidth; ii < endi; ii = ii + 3)
    {

        resBuf[ii] = im2col_getValue(num, rCol);
        resBuf[ii + 1] = im2col_getValue(num, gCol);
        resBuf[ii + 2] = im2col_getValue(num, bCol);
        num++;
    }


    return 0;
}




int main(int argc, char *argv[])
{
    BITMAPFILEHEADER BmpHead;
    BITMAPINFODEADER BmpInfo;

    int BmpWidth;   //图像的宽
    int BmpHeight;  //图像的高
    int BiBitCount; //图像类型，每像素位数 8-灰度图 24-彩色图

    FILE *fp = fopen(BMP_FILE_NAME, "rb"); //二进制读方式打开指定的图像文件
    if (fp == 0)
    {
        cerr << "Can not open " << BMP_FILE_NAME << endl;
        return 0;
    }
    //获取位图文件头结构BITMAPFILEHEADER
    fread(&BmpHead, sizeof(BITMAPFILEHEADER), 1, fp);

    //获取图像宽、高、每像素所占位数等信息
    fread(&BmpInfo, sizeof(BITMAPINFOHEADER), 1, fp);

    // 打印一下文件信息
    //showBmpHead(BmpHead);
    // showBmpInforHead(BmpInfo);

    BmpWidth = BmpInfo.biWidth;   //宽度用来计算每行像素的字节数
    BmpHeight = BmpInfo.biHeight; // 像素的行数
    //计算图像每行像素所占的字节数（必须是4的倍数）
    BiBitCount = BmpInfo.biBitCount;

    // 将图片读取到内存中
    readBmp(fp, pBmpBuf, BmpWidth, BmpHeight, BiBitCount);
    // 计算卷积核
    genGsCore();
    // 将多通道转换为单通道数据
    reFormChannel(BmpWidth, BmpHeight);
    im2col(BmpWidth,BmpHeight);
    im2col_core();


    // MPI 并行计算部分
    int size, myrank, source, dest;
    //MPI_Status status;
    double start_time, end_time;

    int pixStep = 3; // 移动一个像素指针移动的字节数

    int start_x, end_x; // 起始的像素点以及计算区域
    int conv_byte_size; // 卷积区域字节数
    Value = BmpWidth + N / 2 + N / 2;

    unsigned char *result = new unsigned char[BmpWidth * 3 * BmpHeight];
    unsigned char *resBuf = NULL;
    resBuf=new unsigned char [BmpWidth * 3 * BmpHeight];
    memset(resBuf,0,BmpWidth * 3 * BmpHeight);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    start_time = MPI_Wtime();

    if (myrank != 0)
    { //非0号进程发送消息
        //公共计算部分
        start_x = myrank * BmpHeight / size;
        end_x = start_x + BmpHeight / size - 1;
        if (myrank == size - 1)
            end_x = BmpHeight - 1;
        im2col_convolution(resBuf,start_x, end_x, BmpWidth);
        dest=0;
        MPI_Send(resBuf, BmpWidth * 3 * BmpHeight, MPI_UNSIGNED_CHAR, dest, 99, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        printf("%d process :time used to be: %1.2f\n",myrank, end_time-start_time);
    }
    else
    {   // myrank == 0，即0号进程参与计算并负责接受数据
        // 设置参数
        start_x = myrank * BmpHeight / size;
        end_x = start_x + BmpHeight / size - 1;
        if (myrank == size - 1)
            end_x = BmpHeight - 1;
        im2col_convolution(resBuf,start_x, end_x, BmpWidth);

        memcpy(result, resBuf, BmpHeight / size * BmpWidth * 3);
        delete resBuf;
        resBuf = new unsigned char[BmpWidth * 3 * BmpHeight];
        // 合并结果
        for (source = 1; source < size; source++)
        {
            MPI_Recv(resBuf, BmpWidth * 3 * BmpHeight, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &status);
            if(status.MPI_SOURCE!=size-1)
                memcpy(result+status.MPI_SOURCE* BmpHeight / size*BmpWidth*3,resBuf+status.MPI_SOURCE* BmpHeight / size*BmpWidth*3, BmpHeight / size*BmpWidth*3 );
            else
                memcpy(result+status.MPI_SOURCE* BmpHeight / size*BmpWidth*3,resBuf+status.MPI_SOURCE* BmpHeight / size*BmpWidth*3, BmpWidth * 3 * BmpHeight-status.MPI_SOURCE* BmpHeight / size*BmpWidth*3 );
            
        }
        end_time = MPI_Wtime();
        saveBmp("origin.bmp", result, BmpWidth, BmpHeight, BiBitCount);
        printf("All time used to be: %1.2f\n", end_time-start_time);
    }

END:
    MPI_Finalize();
    // MPI End

    delete rBmpBuf;
    delete gBmpBuf;
    delete bBmpBuf;
    delete result;
    delete resBuf;
    delete []rCol;
    delete []gCol;
    delete []bCol;
    if (pBmpBuf)
         delete pBmpBuf;
    fclose(fp);

    return 0;
}