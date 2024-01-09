# LeNet-5 CNN Implementation on GPU Using CUDA

## Introduction

This document serves as a comprehensive guide to the CUDA implementation of the LeNet-5 Convolutional Neural Network (CNN). We cover the initialization of matrices, 2D convolution, bias addition, the hyperbolic tangent activation function, and 2x2 average pooling. Both CPU and GPU versions are included to showcase the performance benefits of utilizing CUDA for deep learning tasks.

## Initialization

## CUDA Grid and Block Dimensions in Matrix Operations

In CUDA, kernels are executed by an array of threads. Threads are organized into blocks, and blocks are organized into a grid. When performing matrix operations, these threads can be allocated to compute different elements of the matrix in parallel.

- **Block Dimensions (`blockDim`)**: This defines the shape of the thread block. For example, if a block dimension is set to `(16, 16, 1)`, each block contains 256 threads in a 16x16 grid. This is useful when the computation can be divided into chunks that fit into this shape, such as processing a 16x16 sub-matrix at a time.

- **Grid Dimensions (`gridDim`)**: This specifies the number of blocks in the grid. If we have a large matrix and a block can compute a 16x16 sub-matrix, the grid dimensions will determine how many such blocks are needed to cover the entire matrix.

In the context of matrix manipulation, each thread might be responsible for computing one element of the output matrix, and the grid and block dimensions will be set accordingly to cover all the elements.

For example, if we have an input matrix of size `1024x1024` and we use blocks of `16x16`, we will need a grid of `64x64` blocks to cover the whole matrix.

## Linearization of Matrices

Linearization is a technique used to represent multi-dimensional arrays in a one-dimensional block of memory, which is how data is stored in C and CUDA by default. For example, an element in a 2D array `A[i][j]` can be accessed in a linearized form as `A[i * width + j]` where `width` is the number of columns in the array. This is crucial in CUDA programming because the GPU memory is accessed as a linear array.

## Detailed Function Implementation

In the following sections, each function implemented is explained in detail, focusing on how block and grid dimensions are utilized and how matrices are linearized for computations.

### Matrix and Array Initialization

The `randomMatrixInit` function initializes a matrix with random values. It represents the matrix in a linear form to be compatible with the memory model of CUDA.

```c
void randomMatrixInit(float *M, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Linearizing the indices: M[i][j] -> M[i * p + j]
            M[i * p + j] = (float)rand() / RAND_MAX;
        }
    }
}
```
The code begins with the `%load_ext nvcc_plugin` command, enabling the execution of CUDA code within the Jupyter notebook.

## A Few Tests

The first section of the code demonstrates the capability to execute simple functions on both the CPU and the GPU, printing "Hello, from the CPU!" and "Hello, from the GPU!" accordingly. This ensures that the environment is correctly set up for CUDA operations.

## Vector Addition

A vector addition function is implemented to demonstrate how GPUs can be used to perform arithmetic operations in parallel. This function initializes two arrays with constant values and sums them element-wise on the GPU.

## LeNet-5 Inference in CUDA

### Matrix and Array Initialization

Functions such as `randomMatrixInit`, `random3DArrayInit`, and `random4DArrayInit` are utilized for initializing matrices and arrays with random values. These are essential for generating the weights and biases for the network before training or inference.

### Convolution Operations

The heart of the CNN is the convolution operation. The code includes a CPU-based implementation, `Conv2D`, and a GPU-accelerated version, `cudaConv2D`. These functions perform a valid convolution between an input matrix and a set of kernels to produce feature maps.

### Bias Addition

After convolution, a bias term is typically added to the result. `AddBias` and its GPU counterpart `cudaAddBias` handle this by adding a bias vector to each feature map.

## Activation Function: Hyperbolic Tangent (tanh)

The hyperbolic tangent (tanh) function introduces non-linearity into the network. This is implemented in `Tanh3D` for the CPU and `cudaTanh3D` for the GPU, transforming each element of the input to its hyperbolic tangent.

tanh transforw the summed weighted input from the neuron into the range of [-1, 1]. In the GPU-accelerated version, `cudaTanh3D` applies the activation in parallel across all neurnos, significantly speeding up the network compared to sequential CPU computations. This is done by assigning a unique thread to each neuron, which makes efficient computation in the network.

### Pooling

Pooling reduce the spatial dimensions of feature maps. `AveragePooling2` and `cudaAveragePooling2` perform 2x2 average pooling, which is used in CNNs to downsample the feature maps.

### Utility Functions

Functions like `matrixPrint` and `printChannel` are implemented to output the content of matrices and specific channels, respectively. These are helpful for debugging and verifying the correctness of the operations.

## Analyusis of what has been done

The goal of implementing these fnctions in both CPU and GPU forms is to highlight the computational efficiency and parallel processing capabilities of GPUs. We observe speedup achieved through CUDA's parallel execution model.

Each function implemented plays a specific role in the CNN's pipeline:

- **Initialization Functions:** Prepare the data structures.
- **Convolution Functions:** Apply filters to the input data to extract features.
- **Bias Addition Functions:** Introduce additionnal parameters to the network that can be learnd during training.
- **Activation Functions:** Introduce non-linear transformation to the network's outputs.
- **Pooling Functions:** Reduce the dimensionality of the feature maps to decrease computational load and potential overfitting.

We can see, by, the way, that the computations have been better on the GPU than CPU as is shown on this plot.
![image](https://github.com/Somekindofa/Lenet_Hardware/assets/93818867/22dce474-c93a-43fd-a974-659f4140c0ad)

By analyzing the performance of these functions, we can conclude about the best practices for GPU programming in the context of deep learning.

## Conclusion

This implementation serves as a foundation for understanding how deep learning can be accelerated using GPUs. It provides a starting point for further exploration into more complex architectures and larger datasets.

