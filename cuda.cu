#include <stdio.h>


__global__ void cube(float *d_in, float *d_out) {

}

int main(int argc, char *argv[]) {
  const int ARRAY_COUNT = 10000;
  const int ARRAY_BYTES = ARRAY_COUNT * sizeof(float);

  float *h_in = (float *)malloc(ARRAY_BYTES);
  float *h_out = (float *)malloc(ARRAY_BYTES);

  // Init
  for (int i = 0; i < ARRAY_COUNT; i++) {
    h_in[i] = i;
  }

  // Fill in
  for (int i = 0; i < ARRAY_COUNT; i++) {
    float src = h_in[i];
    h_out[i] = src * src * src;
  }

  // Print
  for (int i = 0; i < 20; i++) {
    printf("%f: %f\n", h_in[i], h_out[i]);
  }
  printf("\n\n");
  for (int i = ARRAY_COUNT - 20; i < ARRAY_COUNT; i++) {
    printf("%f: %f\n", h_in[i], h_out[i]);
  }
}