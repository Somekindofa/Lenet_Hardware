#include <stdio.h>
#include <time.h>

void MatrixInit(float *M, int n, int p){
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			*(M+i*p+j)=0;
		}
	}
}

void MatrixPrint(float *M, int n, int p){
	printf("\n");
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			printf("%f\t",*(M+i*p+j));
		}
		printf("\n");
	}
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
	//O(n*p) time complexity
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			*(Mout+i*p+j)=*(M1+i*p+j)+*(M2+i*p+j);
		}
	}
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
	//O(1) time complexity
	//We consider the matrix dimensions as gridDim and blockDim
	//n=gridDimx,p=blockDimx
	//i=blockId.x, j=threadId.x
	//linearize the indexes [i,j] -> i*p+j
	int i=blockIdx.x;
	int j=threadIdx.x;
	*(Mout+i*p+j)=*(M1+i*p+j)+*(M2+i*p+j);
}


void MatrixMult(float *M1, float *M2, float *Mout, int n){
	//O(n^3) time complexity
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			for(int k=0;k<n;k++){
				*(Mout+i*n+j)+=*(M1+i*n+k) * *(M2+k*n+j);
			}
		}
	}
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
	//We consider the matrix dimensions as gridDim and blockDim
	//n=gridDimx=blockDimx
	//i=blockIdx.x, j=threadIdx.x
	//linearize the indexes [i,j] -> i*p+j
	//O(n) time complexity
	int i=blockIdx.x;
	int j=threadIdx.x;
	for(int k=0;k<n;k++){
		*(Mout+i*n+j)+=*(M1+i*n+k) * *(M2+k*n+j);
	}
}

int main(int argc,char** argv){
	FILE *fpt;
	fpt = fopen("times.csv", "w+");
	fprintf(fpt,"matrix_size,delta_us_cpu,delta_us_gpu_without_transfer,delta_us_gpu_with_transfer\n");
	int n_[12]={10,25,50,100,200,300,400,500,600,750,850,1000};
	for(int n_idx=0;n_idx<12;n_idx++){
		int n=n_[n_idx];
		struct timespec start_cpu, end_cpu;
		struct timespec start_gpu_with_transfer, end_gpu_with_transfer;
		struct timespec start_gpu_without_transfer, end_gpu_without_transfer;
		
		//initializing on CPU
		float* M1=(float*)malloc(n*n*sizeof(float));
		float* M2=(float*)malloc(n*n*sizeof(float));
		float* Mout_cpu=(float*)malloc(n*n*sizeof(float));
		float* Mout_gpu=(float*)malloc(n*n*sizeof(float));
		MatrixInit(M1,n,n);
		MatrixInit(M2,n,n);
		MatrixInit(Mout_cpu,n,n);
		MatrixInit(Mout_gpu,n,n);
		for (int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				*(M1+i*n+j)=rand()/RAND_MAX; //random data in M1 and M2
				*(M2+i*n+j)=rand()/RAND_MAX;
			}
		}
		
		//start timer CPU
		clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
		
		MatrixMult(M1,M2,Mout_cpu,n);
		
		//end timer CPU
		clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
		
		//start timer GPU with data transfer
		clock_gettime(CLOCK_MONOTONIC_RAW, &start_gpu_with_transfer);
		
		//initializing on GPU
		float *d_M1, *d_M2, *d_Mout;
		cudaMalloc(&d_M1, n*n*sizeof(float));
	    	cudaMalloc(&d_M2, n*n*sizeof(float));
	    	cudaMalloc(&d_Mout, n*n*sizeof(float));
	    	
	    	cudaMemcpy(d_M1, M1, n*n*sizeof(float), cudaMemcpyHostToDevice);
	    	cudaMemcpy(d_M2, M2, n*n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Mout, Mout_gpu, n*n*sizeof(float), cudaMemcpyHostToDevice);
		
		//Processing on GPU
		//start timer GPU without data transfer
		clock_gettime(CLOCK_MONOTONIC_RAW, &start_gpu_without_transfer);
		
		cudaMatrixMult<<<n,n>>>(d_M1,d_M2,d_Mout,n);
		
		//end timer GPU without data transfer
		clock_gettime(CLOCK_MONOTONIC_RAW, &end_gpu_without_transfer);
		
		//Copying result to CPU
		cudaMemcpy(Mout_gpu, d_Mout, n*n*sizeof(float), cudaMemcpyDeviceToHost);
		
		//end timer GPU with data transfer
		clock_gettime(CLOCK_MONOTONIC_RAW, &end_gpu_with_transfer);
		
		int delta_us_cpu = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000000 + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1000;
		int delta_us_gpu_with_transfer = (end_gpu_with_transfer.tv_sec - start_gpu_with_transfer.tv_sec) * 1000000 + (end_gpu_with_transfer.tv_nsec - start_gpu_with_transfer.tv_nsec) / 1000;
		int delta_us_gpu_without_transfer = (end_gpu_without_transfer.tv_sec - start_gpu_without_transfer.tv_sec) * 1000000 + (end_gpu_without_transfer.tv_nsec - start_gpu_without_transfer.tv_nsec) / 1000;
		printf("Matrix size: %d\n",n);
		printf("CPU time (microseconds): %d\n",delta_us_cpu);
		printf("GPU time without data transfer (microseconds): %d\n",delta_us_gpu_without_transfer);
		printf("GPU time with data transfer (microseconds): %d\n\n",delta_us_gpu_with_transfer);
		fprintf(fpt,"%d,%d,%d,%d\n",n,delta_us_cpu,delta_us_gpu_without_transfer,delta_us_gpu_with_transfer);
	}
	fclose(fpt);
	return 0;
}


