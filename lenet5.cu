#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h> //cuda math library
void randomMatrixInit(float *M, int n, int p){
/*
Initializes matrix M of size n*p with values uniformly sampled between 0 and 1.

Inputs:
	-M: Matrix initialized by dynamic memory allocation (float*) M = (float*) malloc(n*p*sizeof(float))
	-n: number of lines of M
	-p: number of columns of M
*/
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			//M[i][j] -> *(M+i*p+j) linearizing the indices
			M[i*p+j ]=(float) rand()/RAND_MAX;
		}
	}
}

void random3DArrayInit(float *A, int n_channels, int n, int p){
/*
Initializes matrix A of size n_channels*n*p with values uniformly sampled between 0 and 1.

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_channels*n*p*sizeof(float))
	-n_channels: number of channels of A
	-n: number of lines of each channel of A
	-p: number of columns of each channel of A
*/
	for (int c=0;c<n_channels;c++){
		for (int i=0;i<n;i++){
			for(int j=0;j<p;j++){
				//A[c][i][j] -> A[c*n*n+i*p+j] linearizing the indices
				A[c*n*n+i*p+j]=(float) rand()/RAND_MAX;
			}
		}
	}
}

void random4DArrayInit(float *A, int n_in_channels, int n_out_channels,int n, int p){
/*
Initializes array A of size n_out_channels*n_in_channels*n*p with values uniformly sampled between 0 and 1.

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_out_channels*n_in_channels*n*p*sizeof(float))
	-n_in_channels: number of input channels of A
	-n_out_channels: number of output channels of A
	-n: number of lines of each channel of A
	-p: number of columns of each channel of A
*/
	for (int c_out=0;c_out<n_out_channels;c_out++){
		for (int c_in=0;c_in<n_in_channels;c_in++){
			for (int i=0;i<n;i++){
				for(int j=0;j<p;j++){
					//A[c][i][j] -> A[c*n*n+i*p+j] linearizing the indices
					A[c_out*n_in_channels*n*n+c_in*n*n+i*p+j]=(float) rand()/RAND_MAX;
				}
			}
		}
	}
}

void zero3DArrayInit(float *A, int n_channels, int n, int p){
/*
Initializes matrix A of size n_channels*n*p with zeros.

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_channels*n*p*sizeof(float))
	-n_channels: number of channels of A
	-n: number of lines of each channel of A
	-p: number of columns of each channel of A
*/
	for (int c=0;c<n_channels;c++){
		for (int i=0;i<n;i++){
			for(int j=0;j<p;j++){
				//A[c][i][j] -> *(A+c*n^2+i*p+j) linearizing the indices
				A[c*n*n+i*p+j]=0;
			}
		}
	}
}

void AddBias(float *A, float* Bias, float* Output, int n_channels, int Mat_size){
/*
Adds Bias to 3D array A. The bias is added as a matrix where each element is the same

Inputs:
	-A: 3D array of shape n_channels*Mat_size*Mat_size
	-Bias: 1D Vector of length n_channels
	-Output: the output array of shape n_channels*Mat_size*Mat_size, allocated with malloc
	-Mat_size: Size of each channel of A
	-n_channels: Number of channels of A
*/
	for(int channel_no=0;channel_no<n_channels;channel_no++){
		float b=Bias[channel_no];
		for(int i=0;i<Mat_size;i++){
			for(int j=0;j<Mat_size;j++){
				int index=channel_no*Mat_size*Mat_size+i*Mat_size+j;
				Output[index]=A[index]+b;
			}
		}
	}
}

__global__ void cudaAddBias(float *A, float* Bias, float* Output, int n_channels, int Mat_size){
/*
Adds Bias to 3D array A. The bias is added as a matrix where each element is the same

Inputs:
	-A: 3D array of shape n_channels*Mat_size*Mat_size
	-Bias: 1D Vector of length n_channels
	-Output: the output array of shape n_channels*Mat_size*Mat_size, allocated with malloc
	-Mat_size: Size of each channel of A
	-n_channels: Number of channels of A

Each thread takes care of one addition, we assume:
	-gridDim.x=n_channels
	-gridDim.y=1
	-gridDim.z=1
	-blockDim.x=Mat_size
	-blockDim.y=Mat_size
	-blockDim.z=1

Thus:
	-channel_no=blockIdx.x
	-i=threadIdx.x
	-j=threadIdx.y
*/
	int channel_no=blockIdx.x;
	int i=threadIdx.x;
	int j=threadIdx.y;
	int index=channel_no*Mat_size*Mat_size+i*Mat_size+j;
	Output[index]=A[index]+Bias[channel_no];
}

void Conv2D(float *Mat, float *Kernels, float *Output, int n_in_channels, int n_out_channels, int Mat_size, int Kernel_size){
/*
Returns the *valid* convolution with stride=1, padding=1 between Matrix and Kernel, computed on CPU.

Inputs:
	-Mat: Input square matrix of size no_input_channels*Mat_size*Mat_size (no depth!)
	initialized by dynamic memory allocation (float*) Mat = (float*) malloc(Mat_size*Mat_size*sizeof(float))
	-Kernels: Input square convolution kernel of size n_out_channels*n_in_channels*Kernel_size*Kernel_size, initalized in the same way
	-Output: Output array of shape no_output_channels*(Mat_size-Kernel_size+1)*(Mat_size-Kernel_size+1), initalized in the same way
	-n_in_channels: Number of input channels
	-n_out_channels: Number of output channels
	-Mat_size: Matrix size
	-Kernel_size: Kernel size
*/
	int Output_size = Mat_size - Kernel_size +1;
	for (int in_channel_no=0;in_channel_no<n_in_channels;in_channel_no++){
		for (int out_channel_no=0;out_channel_no<n_out_channels;out_channel_no++){
			for (int horizontal_block_id=0; horizontal_block_id<Output_size;horizontal_block_id++){
				for (int vertical_block_id=0;vertical_block_id<Output_size;vertical_block_id++){
					float block_output=0;
					for (int i=0;i<Kernel_size;i++){
						for (int j=0;j<Kernel_size;j++){
							//the index of the Matrix with the right channel number, the right vertical and horizontal block and right i and j indexes inside the block
							int mat_index=in_channel_no*Mat_size*Mat_size+(horizontal_block_id+i)*Mat_size+vertical_block_id+j;
							//the index of the kernel with the right input and output channel index and the right position (i,j) inside the kernel
							int kernel_index=out_channel_no*n_in_channels*Kernel_size*Kernel_size + in_channel_no*Kernel_size*Kernel_size + i*Kernel_size+j;
							block_output=block_output+Mat[mat_index] * Kernels[kernel_index];
						}
					}
					int output_index=out_channel_no*Output_size*Output_size + horizontal_block_id*Output_size + vertical_block_id;
					Output[output_index]+=block_output;
				}
			}
		}
	}
}

__global__ void cudaConv2D(float *Mat, float *Kernels, float *Output, int n_in_channels, int n_out_channels, int Mat_size, int Kernel_size) {
/*
Returns the *valid* convolution with stride=1, padding=1 between Matrix (square) and Kernels (square), computed on GPU using CUDA.
We assume:
	-gridDim.x=Output_size
	-gridDim.y=Output_size
	-gridDim.z=n_out_channels
	-blockDim.x=Kernel_size
	-blockDim.y=Kernel_size
	-blockDim.z=n_in_channels
	-Dynamic memory allocation of 2*n_in_channels*Kernel_size*Kernel_size*sizeof(float) (third argument in the <<< >>> brackets) to share block weights for all channels

We have:
	-in_channel_no=
	-out_channel_no=threadIdx.x
	-horizontal_block_id=threadIdx.x
	-vertical_block_id=threadIdx.y
	
Each thread block computes a convolution block for all input and output channel pairs and aggregates them.

Inputs:
	-Mat: Input square matrix of size no_in_channels*Mat_size*Mat_size
	initialized by dynamic memory allocation (float*) Mat = (float*) malloc(no_in_channels*Mat_size*Mat_size*sizeof(float))
	-Kernels: Input square convolution kernel of size n_out_channels*n_in_chanels*Kernel_size*Kernel_size, initalized in the same way
	-Output: Output array of shape no_out_channels*Output_size*Output_size, initalized in the same way
	-n_in_channels: Number of input channels
	-n_out_channels: Number of output channels
	-Mat_size: Matrix size
	-Kernel_size: Kernel size
*/

    int Output_size = Mat_size - Kernel_size + 1;
    int horizontal_block_id = blockIdx.x;
    int vertical_block_id = blockIdx.y;
    int in_channel_no = blockIdx.z;
    int out_channel_no = threadIdx.x;

    // Dynamic shared memory allocation (https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
    //The memory is shared across a block of threads.
    extern __shared__ float shared_memory[]; //size is declared at run time. Must be equal to 2*Kernel_size*Kernel_size*sizeof(float)
    float* Mat_shared=&shared_memory[0];
    float* Kernels_shared=&shared_memory[Kernel_size*Kernel_size];

    float block_output = 0;
    int output_index = out_channel_no * Output_size * Output_size + horizontal_block_id * Output_size + vertical_block_id;

    // Load data into shared memory
    for (int i = 0; i < Kernel_size; i++) {
        for (int j = 0; j < Kernel_size; j++) {
			int mat_index=in_channel_no*Mat_size*Mat_size+(horizontal_block_id+i)*Mat_size+vertical_block_id+j;
			int kernel_index=out_channel_no*n_in_channels*Kernel_size*Kernel_size + in_channel_no*Kernel_size*Kernel_size + i*Kernel_size+j;
        	Mat_shared[i*Kernel_size+j]=Mat[mat_index];
        	Kernels_shared[i*Kernel_size+j]=Kernels[kernel_index];
        }
    }
    // Synchronize threads before using shared memory
    __syncthreads();

    // Compute convolution
    for (int i = 0; i < Kernel_size; i++) {
        for (int j = 0; j < Kernel_size; j++) {
            float m=Mat_shared[i*Kernel_size+j];
            float k=Kernels_shared[i*Kernel_size+j];
            block_output = block_output + m*k;
        }
    }
    Output[output_index] += block_output;
}

void Tanh3D(float* A, float* Output, int n_channels, int Mat_size){
/*
Returns the tanh of input 3D array A.

Inputs:
	-A: Input array of size no_channels*Mat_size*Mat_size
	-Output: Output array of size no_channels*Mat_size*Mat_size
	-no_channels: the number of channels
	-mat_size: the size of the matrix
*/
	for(int channel_no=0;channel_no<n_channels;channel_no++){
		for(int i=0;i<Mat_size;i++){
			for(int j=0;j<Mat_size;j++){
				int index=channel_no*Mat_size*Mat_size+i*Mat_size+j;
				Output[index]=tanh(A[index]);
			}
		}
	}
}

__global__ void cudaTanh3D(float* A,float* Output,int n_channels, int Mat_size){
/*
Returns the tanh of input 3D array A.

We assume:
	-gridDim.x=no_channels
	-blockDim.x=Output_size
	-blockDim.y=Output_size
	
Inputs:
	-A: Input array of size no_channels*Mat_size*Mat_size
	-Output: Output array of size no_channels*Mat_size*Mat_size
	-no_channels: the number of channels
	-mat_size: the size of the matrix
*/
	int channel_no=blockIdx.x;
	int i=threadIdx.x;
	int j=threadIdx.y;
	int index=channel_no*Mat_size*Mat_size+i*Mat_size+j;
	Output[index]=tanh(A[index]);
}

void AveragePooling2(float *A, float *Output, int n_channels, int A_size){
/*
Average pooling of the input matrix by a factor of two in both dimensions, using CPU.

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_channels*n*p*sizeof(float))
	-Output: Output array, of size n_channels*(A_size/2)*(A_size/2)
	-A_size: number of lines and columns of each channel of A, e.g 28 in the Lenet5 example.
	Each channel is assumed to be a square matrix.
	-n_channels: number of channels of A and Output, e.g 6 in the Lenet5 example
	
Example:
	-If Mat is of size 6*28*28 (Lenet5), output is of size 6*14*14, with averaging of each 2*2 block.
*/

	int Output_size = A_size/2;
	for (int channel_no=0; channel_no<n_channels;channel_no++){
		for (int horizontal_block_id=0; horizontal_block_id<Output_size;horizontal_block_id=horizontal_block_id+1){
			for (int vertical_block_id=0;vertical_block_id<Output_size;vertical_block_id=vertical_block_id+1){
			
				//compute sum over 4*4 block
				float block_output=0; //sum of 4*4 block
				for (int i=0;i<2;i++){
					for (int j=0;j<2;j++){
						block_output=block_output + *(A+ channel_no*A_size*A_size + (2*horizontal_block_id+i)*A_size + 2*vertical_block_id+j);
					}
				}
				int output_index=channel_no*Output_size*Output_size + horizontal_block_id*Output_size + vertical_block_id;
				Output[output_index]=block_output/4;
			}
		}
	}
}

__global__ void cudaAveragePooling2(float *A, float *Output, int n_channels, int A_size){
/*
Average pooling of the input matrix by a factor of two in both dimensions, using CUDA on GPU.
We assume n_channels=gridDim.x Output_size=blockDim.x and Output_size=blockDim.y
We have channel_no=blockIdx.x horizontal_block_id=threadIdx.x and vertical_block_id=threadIdx.y
Each thread computes a block

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_channels*n*p*sizeof(float))
	-Output: Output array, of size n_channels*(A_size/2)*(A_size/2)
	-n_channels: number of channels of A and Output, e.g 6 in the Lenet5 example
	-A_size: number of lines and columns of each channel of A, e.g 28 in the Lenet5 example.
	Each channel is assumed to be a square matrix.
	
Example:
	-If Mat is of size 6*28*28 (Lenet5), output is of size 6*14*14, with averaging of each 2*2 block.
*/

	int Output_size = A_size/2;
	int channel_no = blockIdx.x;
	int horizontal_block_id = threadIdx.x;
	int vertical_block_id = threadIdx.y;
	int output_index = channel_no*Output_size*Output_size + horizontal_block_id*Output_size + vertical_block_id;
	
	//compute sum over 4*4 block
	float block_output=0; //sum of 4*4 block
	for (int i=0;i<2;i++){
		for (int j=0;j<2;j++){
			block_output=block_output + *(A+ channel_no*A_size*A_size + (2*horizontal_block_id+i)*A_size + 2*vertical_block_id+j);
		}
	}
	Output[output_index]=block_output/4;
}

// __device__ ​ double tanh ( double  x ) is already defined in CUDA toolkit.
// we just need to cast it to float
__device__ float activation_tanh(float M){
	return (float) tanh((double) M);
}

void matrixPrint(float *M, int n, int p){
/*
Prints the content of matrix M.

Inputs:
	-M: Matrix initialized by dynamic memory allocation (float*) M = (float*) malloc(n*p*sizeof(float))
	-n: number of lines of M
	-p: number of columns of M
*/
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			printf("%f\t",*(M+i*p+j));
		}
		printf("\n");
	}
}

void printChannel(float *A, int channel_no, int n, int p){
/*
Prints the content of channel channel_no of array A.

Inputs:
	-A: Array initialized by dynamic memory allocation (float*) A = (float*) malloc(n_channels*n*p*sizeof(float))
	-channel_no: number of the channel to display, 0 < channel_no < n_channels-1.
	-n: number of lines of each channel of A
	-p: number of columns of each channel of A
*/
	int counter=0;
	for (int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			printf("%f\t",*(A+channel_no*n*p+i*p+j));
			counter++;
		}
		printf("\n");
	}
}

int main(int argc, char** argv){
	//float* shared_memory_cpu=(float*)malloc(2*2*2*sizeof(float)); //2*Kernel_size*Kernel_size for array C as Kernels (2*2*2)
	//float* shared_memory;
	//cudaMalloc(&shared_memory, 2*2*2*sizeof(float));
	//cudaMemcpy(shared_memory, shared_memory_cpu, 2*2*2*sizeof(float), cudaMemcpyHostToDevice);

	//Uncomment code below to prove Conv2D, cudaConv2D, AveragePooling2, cudaAveragePooling2 are functional
	float* A0=(float*)malloc(1*1*4*4*sizeof(float));
	float* A=(float*)malloc(2*4*4*sizeof(float));
	float* B=(float*)malloc(1*1*2*2*sizeof(float));
	float* C=(float*)malloc(2*2*2*2*sizeof(float));
	float* bias=(float*)malloc(2*sizeof(float));
	*A0=1;*(A0+1)=2;*(A0+2)=3;*(A0+3)=4;*(A0+4)=5;*(A0+5)=6;*(A0+6)=7;*(A0+7)=8;*(A0+8)=9;*(A0+9)=10;*(A0+10)=11;*(A0+11)=12;*(A0+12)=13;*(A0+13)=14;*(A0+14)=15;*(A0+15)=16;
	*A=1;*(A+1)=2;*(A+2)=3;*(A+3)=4;*(A+4)=5;*(A+5)=6;*(A+6)=7;*(A+7)=8;*(A+8)=9;*(A+9)=10;*(A+10)=11;*(A+11)=12;*(A+12)=13;*(A+13)=14;*(A+14)=15;*(A+15)=16;
	*(A+16)=1;*(A+17)=0;*(A+18)=0;*(A+19)=0;*(A+20)=0;*(A+21)=1;*(A+22)=0;*(A+23)=0;*(A+24)=0;*(A+25)=0;*(A+26)=1;*(A+27)=0;*(A+28)=0;*(A+29)=0;*(A+30)=0;*(A+31)=1;
	*B=1;*(B+1)=0;*(B+2)=0;*(B+3)=1;
	*C=1;*(C+1)=0;*(C+2)=0;*(C+3)=1;*(C+4)=2;*(C+5)=0;*(C+6)=0;*(C+7)=2;*(C+8)=2;*(C+9)=0;*(C+10)=0;*(C+11)=2;*(C+12)=1;*(C+13)=0;*(C+14)=0;*(C+15)=1;
	*bias=1;*(bias+1)=2;
	float* out=(float*)malloc(9*sizeof(float)); //A0 conv B on CPU
	float* AcC_cpu=(float*)malloc(18*sizeof(float)); //A conv C on CPU
	float* oout=(float*)malloc(8*sizeof(float)); //subsampled A on CPU
	float* out_from_gpu=(float*)malloc(9*sizeof(float)); //A0 conv B on GPU
	float* oout_from_gpu=(float*)malloc(8*sizeof(float)); //subsampled A on GPU
	float* AcC_from_gpu=(float*)malloc(18*sizeof(float)); //A conv C on GPU
	float* Aplusbias=(float*)malloc(2*4*4*sizeof(float)); //A plus bias on CPU
	float* Aplusbias_from_gpu=(float*)malloc(2*4*4*sizeof(float));
	float* tanhA=(float*)malloc(2*4*4*sizeof(float));
	float* tanhA_from_gpu=(float*)malloc(2*4*4*sizeof(float));
	float *d_A0,*d_A, *d_B,*d_C,*d_out,*d_oout,*d_AcC_gpu,*d_Aplusbias,*d_bias,*d_tanhA;
	cudaMalloc(&d_A0, 16*sizeof(float));
	cudaMalloc(&d_A, 32*sizeof(float));
	cudaMalloc(&d_B, 4*sizeof(float));
	cudaMalloc(&d_C, 16*sizeof(float));
	cudaMalloc(&d_out, 9*sizeof(float));
	cudaMalloc(&d_oout, 8*sizeof(float));
	cudaMalloc(&d_AcC_gpu, 18*sizeof(float));
	cudaMalloc(&d_Aplusbias, 2*4*4*sizeof(float));
	cudaMalloc(&d_tanhA, 2*4*4*sizeof(float));
	cudaMalloc(&d_bias, 2*sizeof(float));
	cudaMemcpy(d_A0, A0, 16*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, A, 32*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, 4*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, 16*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, 9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_oout, oout, 8*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_AcC_gpu, AcC_from_gpu, 18*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Aplusbias, Aplusbias_from_gpu, 2*4*4*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, 2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tanhA, tanhA, 2*4*4*sizeof(float), cudaMemcpyHostToDevice);

	printf("A0[:,:]=\n");
	matrixPrint(A0, 4, 4);
	printf("\nB[:,:]=\n");
	matrixPrint(B, 2, 2);
	printf("\nA[0,:,:]=\n");
	printChannel(A,0,4,4);
	printf("\nA[1,:,:]=\n");
	printChannel(A,1,4,4);
	printf("\nC[0,0,:,:]=\n");
	printChannel(C,0,2,2);
	printf("\nC[0,1,:,:]=\n");
	printChannel(C,1,2,2);
	printf("\nC[1,0,:,:]=\n");
	printChannel(C,2,2,2);
	printf("\nC[1,1,:,:]=\n");
	printChannel(C,3,2,2);
	printf("\nbias=\n");
	matrixPrint(bias,1,2);

	Conv2D(A0,B,out,1,1,4,2);
	Conv2D(A,C,AcC_cpu,2,2,4,2);
	dim3 gridSizeA0(3,3,1);
	dim3 blockSizeA0(1,1,1);
	cudaConv2D<<<gridSizeA0,blockSizeA0,2*2*2*sizeof(float)>>>(d_A0,d_B,d_out,1,1,4,2);
	cudaMemcpy(out_from_gpu, d_out, 9*sizeof(float), cudaMemcpyDeviceToHost);
	dim3 gridSizeA(3,3);
	dim3 blockSizeA(2,2);
	cudaConv2D<<<gridSizeA,blockSizeA,2*2*2*sizeof(float)>>>(d_A,d_C,d_AcC_gpu,2,2,4,2);
	cudaMemcpy(AcC_from_gpu, d_AcC_gpu, 18*sizeof(float), cudaMemcpyDeviceToHost);
	AveragePooling2(A, oout, 2, 4);
	dim3 gridSize(2,1,1);
	dim3 blockSize(4,4,1);
	cudaAveragePooling2<<<gridSize,blockSize>>>(d_A,d_oout,2,4);
	cudaMemcpy(oout_from_gpu, d_oout, 8*sizeof(float), cudaMemcpyDeviceToHost);
	AddBias(A,bias,Aplusbias,2,4);
	cudaAddBias<<<gridSize,blockSize>>>(d_A,d_bias,d_Aplusbias,2,4);
	cudaMemcpy(Aplusbias_from_gpu, d_Aplusbias, 32*sizeof(float), cudaMemcpyDeviceToHost);
	Tanh3D(A,tanhA,2,4);
	cudaTanh3D<<<gridSize,blockSize>>>(d_A,d_tanhA,2,4);
	cudaMemcpy(tanhA_from_gpu, d_tanhA, 32*sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nA0 conv B on CPU:\n");
	matrixPrint(out, 3, 3);
	printf("\nA0 conv B on GPU:\n");
	matrixPrint(out_from_gpu, 3, 3);
	printf("\nA conv C on CPU (2 in channels, 2 out channel) - 1st output channel:\n");
	printChannel(AcC_cpu,0,3,3);
	printf("\nA conv C on CPU (2 in channels, 2 out channel) - 2nd output channel:\n");
	printChannel(AcC_cpu,1,3,3);
	printf("\nA conv C on GPU (2 in channels, 2 out channel) - 1st output channel:\n");
	printChannel(AcC_from_gpu,0,3,3);
	printf("\nA conv C on GPU (2 in channels, 2 out channel) - 2nd output channel:\n");
	printChannel(AcC_from_gpu,1,3,3);
	printf("\n2x2 average pooling of A on CPU:\n");
	printf("\nAp[0,:,:]\n");
	printChannel(oout,0,2,2);
	printf("\nAp[1,:,:]\n");
	printChannel(oout,1,2,2);
	printf("\n2x2 average pooling of A on GPU:\n");
	printf("\nAp[0,:,:]\n");
	printChannel(oout_from_gpu,0,2,2);
	printf("\nAp[1,:,:]\n");
	printChannel(oout_from_gpu,1,2,2);
	printf("\nA plus bias on CPU - 1st channel:\n");
	printChannel(Aplusbias,0,4,4);
	printf("\nA plus bias on CPU - 2nd channel:\n");
	printChannel(Aplusbias,1,4,4);
	printf("\nA plus bias on GPU - 1st channel:\n");
	printChannel(Aplusbias_from_gpu,0,4,4);
	printf("\nA plus bias on GPU - 2nd channel:\n");
	printChannel(Aplusbias_from_gpu,1,4,4);

	//initialize random number generator
	/*
	time_t t;
	srand((unsigned) time(&t));
	
	//initializing 2D and 3D arrays
	int raw_data_n=32;
	float* raw_data=(float*)malloc(raw_data_n*raw_data_n*sizeof(float));
	randomMatrixInit(raw_data, raw_data_n, raw_data_n);
	float* d_raw_data;
	cudaMalloc(&d_raw_data,raw_data_n*raw_data_n*sizeof(float));
	cudaMemcpy(d_raw_data, raw_data, raw_data_n*raw_data_n*sizeof(float), cudaMemcpyHostToDevice);
	
	int C1_data_n=28;
	int C1_data_n_channels=6;
	float* C1_data=(float*) malloc(C1_data_n_channels*C1_data_n*C1_data_n*sizeof(float));
	zero3DArrayInit(C1_data, C1_data_n_channels, C1_data_n, C1_data_n);
	float* d_C1_data;
	cudaMalloc(&d_C1_data,C1_data_n_channels*C1_data_n*C1_data_n*sizeof(float));
	cudaMemcpy(d_C1_data, C1_data, C1_data_n_channels*C1_data_n*C1_data_n*sizeof(float), cudaMemcpyHostToDevice);

	int C1_kernel_n=5;
	int C1_kernel_n_in_channels=1;
	int C1_kernel_n_out_channels=C1_data_n_channels;
	int C1_kernel_size=C1_kernel_n_out_channels*C1_kernel_n_in_channels*C1_kernel_n*C1_kernel_n*sizeof(float);
	float* C1_kernel=(float*)malloc(C1_kernel_size);
	random4DArrayInit(C1_kernel, C1_kernel_n_in_channels, C1_kernel_n_out_channels, C1_kernel_n, C1_kernel_n);
	float* d_C1_kernel;
	cudaMalloc(&d_C1_kernel,C1_kernel_size);
	cudaMemcpy(d_C1_kernel, C1_kernel, C1_kernel_size, cudaMemcpyHostToDevice);
	
	int S2_data_n=C1_data_n/2;
	int S2_data_n_channels=C1_data_n_channels;
	float* S2_data=(float*)malloc(S2_data_n_channels*S2_data_n*S2_data_n*sizeof(float));
	zero3DArrayInit(S2_data, S2_data_n_channels, S2_data_n, S2_data_n);
	float* d_S2_data;
	cudaMalloc(&d_S2_data,S2_data_n_channels*S2_data_n*S2_data_n*sizeof(float));
	cudaMemcpy(d_S2_data, S2_data, S2_data_n_channels*S2_data_n*S2_data_n*sizeof(float), cudaMemcpyHostToDevice);	

	int C3_data_n=10;
	int C3_data_n_channels=16;
	float* C3_data=(float*) malloc(C3_data_n_channels*C3_data_n*C3_data_n*sizeof(float));
	zero3DArrayInit(C3_data, C3_data_n_channels, C3_data_n, C3_data_n);
	float* d_C3_data;
	cudaMalloc(&d_C3_data,C3_data_n_channels*C3_data_n*C3_data_n*sizeof(float));
	cudaMemcpy(d_C3_data, C3_data, C3_data_n_channels*C3_data_n*C3_data_n*sizeof(float), cudaMemcpyHostToDevice);

	int C3_kernel_n=5;
	int C3_kernel_n_in_channels=S2_data_n_channels;
	int C3_kernel_n_out_channels=C3_data_n_channels;
	int C3_kernel_size=C3_kernel_n_out_channels*C3_kernel_n_in_channels*C3_kernel_n*C3_kernel_n*sizeof(float);
	float* C3_kernel=(float*)malloc(C3_kernel_size);
	random4DArrayInit(C3_kernel, C3_kernel_n_in_channels, C3_kernel_n_out_channels, C3_kernel_n, C3_kernel_n);
	float* d_C3_kernel;
	cudaMalloc(&d_C3_kernel,C3_kernel_size);
	cudaMemcpy(d_C3_kernel, C3_kernel, C3_kernel_size, cudaMemcpyHostToDevice);

	int S4_data_n=C3_data_n/2;
	int S4_data_n_channels=C3_data_n_channels;
	float* S4_data=(float*)malloc(S4_data_n_channels*S4_data_n*S4_data_n*sizeof(float));
	zero3DArrayInit(S4_data, S4_data_n_channels, S4_data_n, S4_data_n);
	float* d_S4_data;
	cudaMalloc(&d_S4_data,S4_data_n_channels*S4_data_n*S4_data_n*sizeof(float));
	cudaMemcpy(d_S4_data, S4_data, S4_data_n_channels*S4_data_n*S4_data_n*sizeof(float), cudaMemcpyHostToDevice);	

	//First convolution
	dim3 gridSizeC1(C1_data_n_channels,1,1);
	dim3 blockSizeC1(C1_data_n,C1_data_n,1);
	cudaConv2D<<<gridSizeC1,blockSizeC1>>>(d_raw_data,d_C1_kernel,d_C1_data,1,C1_data_n_channels,raw_data_n,C1_kernel_n);
	cudaMemcpy(C1_data, d_C1_data, C1_data_n_channels*C1_data_n*C1_data_n*sizeof(float), cudaMemcpyDeviceToHost);
	
	//Average pooling
	dim3 gridSize2(C1_data_n_channels,1,1);
	dim3 blockSize2(S2_data_n,S2_data_n,1);
	cudaAveragePooling2<<<gridSize2,blockSize2>>>(d_C1_data,d_S2_data,C1_data_n_channels,C1_data_n);
	cudaMemcpy(S2_data, d_S2_data, C1_data_n_channels*S2_data_n*S2_data_n*sizeof(float), cudaMemcpyDeviceToHost);
	*/
}
