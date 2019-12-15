#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <time.h> 
#include <stdio.h>
#include <stdlib.h>


#define THREAD_COUNT 1024
#define CONVOLUTION_SIZE 3

__global__ void convolutionParallel(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count, int convolution_size)
{
	// process image
	int offset = (blockIdx.x * blockDim.x + threadIdx.x);
	int width_out = (width - convolution_size + 1);
	int height_out = (height - convolution_size + 1);

	//Loop over pixels of smaller image
	for (int i = offset; i < width_out * height_out * 4; i += thread_count)
	{
		int row = i / (4*width_out);
		int col = i % (4*width_out);
		int reference_pixel_offset = 4 * row * width + col;
		float sum = 0.0;
		
		if (convolution_size == 3)
		{
			float w[9] =
			{
			  1,	2,		-1,
			  2,	0.25,	-2,
			  1,	-2,		-1
			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];
		}

		if (convolution_size == 5)
		{
			float w[25] =
			{
				0.5,	0.75,	1,		-0.75,	-0.5,
				0.75,	1,		2,		-1,		-0.75,
				1,		2,		0.25,	-2,		-1,
				0.75,	1,		-2,		-1,		-0.75,
				0.5,	0.75,	-1,		-0.75,	-0.5
			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];

		}
		if (convolution_size == 7)
		{
			float w[49] =
			{
				0.25,	0.3, 	0.5, 	0.75, 	-0.5, 	-0.3, 	-0.25,
				0.3,	0.5,	0.75,	1,		-0.75,	-0.5, 	-0.3,
				0.5,	0.75,	1,		2,		-1,		-0.75,	-0.5,
				0.75,	1,		2,		0.25,	-2,		-1, 	-0.75,
				0.5,	0.75,	1,		-2,		-1,		-0.75, 	-0.5,
				0.3,	0.5,	0.75,	-1,		-0.75,	-0.5, 	-0.3,
				0.25, 	0.3,	0.5,	-0.75,	-0.5, 	-0.3, 	-0.25

			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];
		}

		if (sum <= 0)			sum = 0;
		if (sum >= 255)			sum = 255;
		if ((i + 1) % 4 == 0)	sum = 255; // Set a = 255

		new_image[i] = (int) sum;

	}

}


void convolutionSerial(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count, int convolution_size)
{
	// process image
	int width_out = (width - convolution_size + 1);
	int height_out = (height - convolution_size + 1);

	//Loop over pixels of smaller image
	for (int i = 0; i < width_out * height_out * 4; i += thread_count)
	{
		int row = i / (4 * width_out);
		int col = i % (4 * width_out);
		int reference_pixel_offset = 4 * row * width + col;
		float sum = 0.0;

		if (convolution_size == 3)
		{
			float w[9] =
			{
			  1,	2,		-1,
			  2,	0.25,	-2,
			  1,	-2,		-1
			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];
		}

		if (convolution_size == 5)
		{
			float w[25] =
			{
				0.5,	0.75,	1,		-0.75,	-0.5,
				0.75,	1,		2,		-1,		-0.75,
				1,		2,		0.25,	-2,		-1,
				0.75,	1,		-2,		-1,		-0.75,
				0.5,	0.75,	-1,		-0.75,	-0.5
			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];

		}

		if (convolution_size == 7)
		{
			float w[49] =
			{
				0.25,	0.3, 	0.5, 	0.75, 	-0.5, 	-0.3, 	-0.25,
				0.3,	0.5,	0.75,	1,		-0.75,	-0.5, 	-0.3,
				0.5,	0.75,	1,		2,		-1,		-0.75,	-0.5,
				0.75,	1,		2,		0.25,	-2,		-1, 	-0.75,
				0.5,	0.75,	1,		-2,		-1,		-0.75, 	-0.5,
				0.3,	0.5,	0.75,	-1,		-0.75,	-0.5, 	-0.3,
				0.25, 	0.3,	0.5,	-0.75,	-0.5, 	-0.3, 	-0.25

			};

			for (int j = 0; j < convolution_size; j++)
				for (int k = 0; k < convolution_size; k++)
					sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];

		}

		if (sum <= 0)			sum = 0;
		if (sum >= 255)			sum = 255;
		if ((i + 1) % 4 == 0)	sum = 255; // Set a = 255

		new_image[i] = (int)sum;

	}
}



void executeConvolution()
{

	unsigned error;
	unsigned char* image;
	unsigned char* new_image;
	unsigned int width, height;
	unsigned char* d_image;
	unsigned char* d_new_image;

	int matrix_size_offset = CONVOLUTION_SIZE - 1;

	error = lodepng_decode32_file(&image, &width, &height, "original.png");
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	d_image = (unsigned char*)malloc(height * width * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& d_image, width * height * 4 * sizeof(unsigned char));

	d_new_image = (unsigned char*)malloc((width - matrix_size_offset) * (height - matrix_size_offset) * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& d_new_image, (width - matrix_size_offset) * (height - matrix_size_offset) * 4 * sizeof(unsigned char));


	for (int i = 0; i < height * width * 4; i++) { d_image[i] = image[i]; }

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	/*clock_t start, end;
	double cpu_time_used;

	start = clock();*/
	
	
	//cudaEventRecord(start);
	convolutionParallel << < 1, THREAD_COUNT >> > (d_image, d_new_image, height, width, THREAD_COUNT, CONVOLUTION_SIZE);
	//convolutionSerial(d_image, d_new_image, height, width, 1, CONVOLUTION_SIZE);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//end = clock();
	//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f", milliseconds);
	lodepng_encode32_file("convolutionOut3X3.png", d_new_image, width - matrix_size_offset, height - matrix_size_offset);

	cudaFree(d_image);
	cudaFree(d_new_image);

}

int main()
{
	executeConvolution();
}