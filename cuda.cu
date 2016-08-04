#include <stdio.h>

const int ARRAY_LENGTH = 100000;
const int THREAD_COUNT = 1000;
const int ARRAY_BYTES = ARRAY_LENGTH * sizeof(float);

__global__ void array_init(float *d_in) {
    int idx = blockIdx.x * THREAD_COUNT + threadIdx.x;
    d_in[idx] = idx;
}

__global__ void cube(float *d_in, float *d_out) {
    int idx = blockIdx.x * THREAD_COUNT + threadIdx.x;
    float src = d_in[idx];
    d_out[idx] = src * src * src;
}

int main(int argc, char *argv[]) {

  float *h_out = (float *)malloc(ARRAY_BYTES);

  float *d_in;
  float *d_out;
  cudaMalloc(&d_in, ARRAY_BYTES);
  cudaMalloc(&d_out, ARRAY_BYTES);

  int kBlockCount = ARRAY_LENGTH / THREAD_COUNT;

  array_init<<<kBlockCount,THREAD_COUNT>>>(d_in);

  cube<<<kBlockCount,THREAD_COUNT>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  // Print
  for (int i = 0; i < 10; i++) {
    printf("%.0f\n", h_out[i]);
  }
  printf("\n\n");
  for (int i = ARRAY_LENGTH - 10; i < ARRAY_LENGTH; i++) {
    printf("%.0f\n", h_out[i]);
  }
}

// int main(int argc, char *argv[]) {
//   const int ARRAY_LENGTH = 100000;
//   const int ARRAY_BYTES = ARRAY_LENGTH * sizeof(float);

//   float *h_in = (float *)malloc(ARRAY_BYTES);
//   float *h_out = (float *)malloc(ARRAY_BYTES);

//   // Init
//   for (int i = 0; i < ARRAY_LENGTH; i++) {
//     h_in[i] = i;
//   }

//   // Fill in
//   for (int i = 0; i < ARRAY_LENGTH; i++) {
//     float src = h_in[i];
//     h_out[i] = src * src * src;
//   }

//   // Print
//   for (int i = 0; i < 10; i++) {
//     printf("%.0f: %.0f\n", h_in[i], h_out[i]);
//   }
//   printf("\n\n");
//   for (int i = ARRAY_LENGTH - 10; i < ARRAY_LENGTH; i++) {
//     printf("%.0f: %.0f\n", h_in[i], h_out[i]);
//   }
// }