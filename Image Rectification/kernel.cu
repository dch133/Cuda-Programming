#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#define SIZE 1
#define THREAD_COUNT 1050
#define BLOCK_COUNT 2

// rectify <<< block_count, thread_count >>>
__global__ void rectify(unsigned char* image, unsigned height, unsigned width, int thread_count)
{
	// process image
	int block = (height * width * 4) / thread_count;
	int offset = threadIdx.x * block;
	for (int i = 0; i < block; i++)
	{
		int j = offset + i;
		if (image[j] < 127)	image[j] = 127;

	}

}

// rectify <<< block_count, thread_count >>>
__global__ void pool(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count)
{
	// process image
	int offset = (blockIdx.x * blockDim.x + threadIdx.x)*4;

	for (int i = offset; i < (width*height); i+=(thread_count*4) )
	{
		int x = i % (width * 2) * 2;
		int y = i / (width * 2);
		int p1 = 8 * width * y + x;
		int p2 = 8 * width * y + x + 4;
		int p3 = 8 * width * y + x + 4 *  width;
		int p4 = 8 * width * y + x + 4 * width + 4;

		unsigned r[] = { image[p1],   image[p2],   image[p3],   image[p4] };
		unsigned g[] = { image[p1+1], image[p2+1], image[p3+1], image[p4+1] };
		unsigned b[] = { image[p1+2], image[p2+2], image[p3+2], image[p4+2] };
		unsigned a[] = { image[p1+3], image[p2+3], image[p3+3], image[p4+3] };
			
		int rMax = r[0];
		int gMax = g[0];
		int bMax = b[0];
		int aMax = a[0];
			
		for (int j = 1; j < 4; j++ ) 
		{
			if (r[j] > rMax) rMax = r[j];
			if (g[j] > gMax) gMax = g[j];
			if (b[j] > bMax) bMax = b[j];
			if (a[j] > aMax) aMax = a[j];
			
		}
		new_image[i] = rMax;
		new_image[i+1] = gMax;
		new_image[i+2] = bMax;
		new_image[i+3] = aMax;
		

	}
}

void pool2(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count)
{
	// process image
	int n = 0;

	for (int i = 0; i < width * height* 4;  i +=8) {

		if (i%width * height * 4 == 0 ){
			i += width * 4;

		}
			for (int k = 0; k < 4;k++) {
				// k = color
				// i = offset in image array
				// the rest is for correct pixel in 2X2 block
				int a = image[k + i];
				int b = image[k + i + 4];
				int c = image[k + i + width*4];
				int d = image[k + i + width * 4  + 4];



				 
				int max = a > b ? a : b;
				max = c > max ? c : max;
				max = d > max ? d : max;
				new_image[n++] = max;

			}


		
	}
			
		

	
}



void printArray(unsigned char* a, int n)
{
	for (int i = 0; i < n; i++) { printf("%u ", a[i]); }
}

void executeRectifying()
{
	//timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//get the image
	unsigned error;
	unsigned char* image;
	unsigned int width, height;
	unsigned char* d_image;


	error = lodepng_decode32_file(&image, &width, &height, "test.png");
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	d_image = (unsigned char*)malloc(height * width * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& d_image, width * height * 4 * sizeof(unsigned char));

	for (int i = 0; i < height * width * 4; i++) { d_image[i] = image[i]; }
	//printArray(d_image, 100);
	int thread_count = THREAD_COUNT;
	if (thread_count > 1024) thread_count = 1024;


	cudaEventRecord(start);
	rectify <<< 1,thread_count>>> (d_image, height, width, thread_count);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("%f", milliseconds);

	lodepng_encode32_file("rectifyOut.png", d_image, width, height);
	cudaFree(d_image);
}

void executePooling()
{
	////timer
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	unsigned error;
	unsigned char* image;
	unsigned char* new_image;
	unsigned int width, height;
	unsigned char* d_image;
	unsigned char* d_new_image;

	error = lodepng_decode32_file(&image, &width, &height, "test.png");
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	
	d_image = (unsigned char*)malloc(height * width * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& d_image, width * height * 4 * sizeof(unsigned char));

	d_new_image = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	cudaMallocManaged((void**)& d_new_image, width * height * sizeof(unsigned char));


	for (int i = 0; i < height * width * 4; i++) { d_image[i] = image[i]; }

	//pool <<< 1, THREAD_COUNT >> > (d_image, d_new_image, height, width, THREAD_COUNT);
	if (THREAD_COUNT == 1) {
		pool2(d_image, d_new_image, height, width, THREAD_COUNT);

	}
	else {
		pool <<< BLOCK_COUNT, THREAD_COUNT/ BLOCK_COUNT >>> (d_image, d_new_image, height, width, THREAD_COUNT);

	}
	// <blockNum,blockNum/threadcount>
	cudaDeviceSynchronize();
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);

	lodepng_encode32_file("poolOut.png", d_new_image, width/2, height/2);

	cudaFree(d_image);
	cudaFree(d_new_image);

}



//int main(int argc, char* argv[])
//{
//
//	//executeRectifying();
//	executePooling();
//
//	return 0;
//}

