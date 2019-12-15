
//Q2 ------------------------------------------


#include"stdio.h"
#include"math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "A_10.h"
#include "B_10.h"
#include "A_1024.h"
#include "B_1024.h"
#include "X_1024.h"


#define THREAD_COUNT 32

const int MATRIX_SIZE = 1024;




__global__ void matrixmul(double *a, double *b, double *c, int n)
{
	//starting offset for current thread
	int offset = threadIdx.x *  n / THREAD_COUNT;
	//stop processing cell once youve processed n / number of threads
	int limit = offset + n / THREAD_COUNT;
	

	if (threadIdx.x < n) {
		//each threads takes care of a limited number of cells ( n / number of threads) 
		for (offset; offset < limit; offset++) {
			double sum = 0;
			//iterate over for each cell, all over corresponding sum of product
			for (int i = 0; i < n; i++) {
				sum += a[offset * n + i] * b[i];
			}
			c[offset] = sum;
		}
	}

}


int main()
{
//	//MATRIX INVERSION ---------------------------------
//	double** A, ** I, temp;
//	int i, j, k;
//
//	I = (double**)malloc(MATRIX_SIZE * sizeof(double*));
//	for (i = 0; i < MATRIX_SIZE; i++) {
//	I[i] = (double*)malloc(MATRIX_SIZE * sizeof(double));
//}
//	A = (double**)malloc(MATRIX_SIZE * sizeof(double*));
//	for (i = 0; i < MATRIX_SIZE; i++) {
//	A[i] = (double*)malloc(MATRIX_SIZE * sizeof(double));
//}
//
//	for (i = 0; i < MATRIX_SIZE; i++) {
//		for (j = 0; j < MATRIX_SIZE; j++) {
//			A[i][j] = A_10[i][j];
//		}
//	}
//
//	for (i = 0; i < MATRIX_SIZE; i++) {
//		for (j = 0; j < MATRIX_SIZE; j++) {
//			if (i == j)
//				I[i][j] = 1;
//			else
//				I[i][j] = 0;
//		}
//	}
//	/*---------------LoGiC starts here------------------*/
//	for (k = 0; k < MATRIX_SIZE; k++)
//	{
//		temp = A[k][k];
//		for (j = 0; j < MATRIX_SIZE; j++)
//		{
//			A[k][j] /= temp;
//			I[k][j] /= temp;
//		}													//R1=R1-R0*A[1][0] similarly for I
//		for (i = 0; i < MATRIX_SIZE; i++)								//R2=R2-R0*A[2][0]		,,
//		{
//			temp = A[i][k];									//R1=R1/A[1][1]
//			for (j = 0; j < MATRIX_SIZE; j++)							//R0=R0-R1*A[0][1]
//			{												//R2=R2-R1*A[2][1]
//				if (i == k)
//					break;									//R2=R2/A[2][2]
//				A[i][j] -= A[k][j] * temp;						//R0=R0-R2*A[0][2]
//				I[i][j] -= I[k][j] * temp;						//R1=R1-R2*A[1][2]
//			}
//		}
//	}

	const int size_of_b = 1;
	printf("test matrix mult \n");
	double* inverseMat = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	double* bMat = (double*)malloc(MATRIX_SIZE * sizeof(double));

	/*---------------LoGiC ends here--------------------*/
	/*printf("The inverse of the matrix is:				");
	for (i = 0; i < MATRIX_SIZE; i++)
	{
		for (j = 0; j < MATRIX_SIZE; j++) {
			inverseMat[i + j * MATRIX_SIZE] = I[i][j];
			printf("%f ", I[i][j]);
		}
		printf("\n");

	}*/


	//MATRIX MULTIPLICATION ----------------------------




	double* d_1 = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocManaged((void**)& d_1, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));

	double* d_2 = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocManaged((void**)& d_2, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));

	double* d_3 = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocManaged((void**)& d_3, MATRIX_SIZE *MATRIX_SIZE * sizeof(double));


	for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE ; i++) {

		d_1[i] = A_1024[i];

	}
	for (int i = 0; i < MATRIX_SIZE ; i++) {

		d_2[i] = X_1024[i];
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	//BLOCK_COUNT, THREAD_COUNT/ BLOCK_COUNT 
	matrixmul << <1, THREAD_COUNT >> > (d_1, d_2, d_3,MATRIX_SIZE);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaThreadSynchronize();



	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time : %f \n", milliseconds);



	for (int i = 0; i <MATRIX_SIZE; i++)
	{

		printf("%f   ", d_3[i]);


		printf("\n");
	}



}



