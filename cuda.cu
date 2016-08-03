#include <stdio.h>


__global__ void array_init(float *d_in) {
    d_in[threadIdx.x] = threadIdx.x;
}

__global__ void cube(float *d_in, float *d_out) {
    float src = d_in[threadIdx.x];
    d_out[threadIdx.x] = src * src * src;
}

int main(int argc, char *argv[]) {
  const int ARRAY_COUNT = 100000;
  const int ARRAY_BYTES = ARRAY_COUNT * sizeof(float);

  float *h_out = (float *)malloc(ARRAY_BYTES);

  float *d_in;
  float *d_out;
  cudaMalloc(&d_in, ARRAY_BYTES);
  cudaMalloc(&d_out, ARRAY_BYTES);

  array_init<<<1,1000>>>(d_in);

  cube<<<1,1000>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  // Print
  for (int i = 0; i < 10; i++) {
    printf("%.0f\n", h_out[i]);
  }
  printf("\n\n");
  for (int i = ARRAY_COUNT - 10; i < ARRAY_COUNT; i++) {
    printf("%.0f\n", h_out[i]);
  }
}

// int main(int argc, char *argv[]) {
//   const int ARRAY_COUNT = 100000;
//   const int ARRAY_BYTES = ARRAY_COUNT * sizeof(float);

//   float *h_in = (float *)malloc(ARRAY_BYTES);
//   float *h_out = (float *)malloc(ARRAY_BYTES);

//   // Init
//   for (int i = 0; i < ARRAY_COUNT; i++) {
//     h_in[i] = i;
//   }

//   // Fill in
//   for (int i = 0; i < ARRAY_COUNT; i++) {
//     float src = h_in[i];
//     h_out[i] = src * src * src;
//   }

//   // Print
//   for (int i = 0; i < 10; i++) {
//     printf("%.0f: %.0f\n", h_in[i], h_out[i]);
//   }
//   printf("\n\n");
//   for (int i = ARRAY_COUNT - 10; i < ARRAY_COUNT; i++) {
//     printf("%.0f: %.0f\n", h_in[i], h_out[i]);
//   }
// }