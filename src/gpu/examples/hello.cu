#include <stdio.h>


__global__ void kernel() {
}

int main() {
    kernel<<<1, 1>>>();
    printf("hello, world!\n");

    return 0;
}