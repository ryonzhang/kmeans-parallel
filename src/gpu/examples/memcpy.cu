#include <cuda.h>
#include <stdio.h>


#define SIZE 4


int *d_arr;
int *h_arr;


__device__ void device(int *d_arr) {
    printf("%p\n", d_arr);
}

__global__ void kernel(int *d_arr) {
    device(d_arr);
}

__host__ int main () {
    h_arr = (int *) malloc(SIZE);
    cudaMalloc((void **) &d_arr, SIZE);
    cudaMemcpy(d_arr, h_arr, SIZE, cudaMemcpyHostToDevice);
    
    kernel<<<1, 1>>>(d_arr);

    free(h_arr);
    cudaFree(d_arr);
    return 0;
}