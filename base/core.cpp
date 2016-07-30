#include <string.h>
#include <math.h>

#include "core.h"
#include "base.h"

inline void DrawPixel(pixel_buffer *PixelBuffer, int X, int Y, u32 Color) {
  if (X < 0 || X > PixelBuffer->width || Y < 0 || Y > PixelBuffer->height) {
    return;
  }
  Y = PixelBuffer->height - Y;  // Origin in bottom-left
  u32 *pixel = (u32 *)PixelBuffer->memory + X + Y * PixelBuffer->width;
  *pixel = Color;
}

struct v2i {
  int x;
  int y;
};

struct v3f {
  r32 x;
  r32 y;
  r32 z;
};

void DrawLine(pixel_buffer *PixelBuffer, v2i A, v2i B, u32 Color) {
  bool swapped = false;
  if (abs(B.x - A.x) < abs(B.y - A.y)) {
    // Make y the driving axis
    int tmp = A.x;
    A.x = A.y;
    A.y = tmp;
    tmp = B.x;
    B.x = B.y;
    B.y = tmp;
    swapped = true;
  }
  if (B.x - A.x < 0) {
    // A should always be on the "left"
    v2i tmp = A;
    A = B;
    B = tmp;
  }

  r32 m = (r32)(B.y - A.y) / (B.x - A.x);
  r32 error = m - 1;
  int y = A.y;
  for (int x = A.x; x <= B.x; x++) {
    if (!swapped) {
      DrawPixel(PixelBuffer, x, y, Color);
    } else {
      DrawPixel(PixelBuffer, y, x, Color);
    }
    error += m;
    if (error > 0) {
      error -= 1;
      y++;
    }
  }
}

r32 func(r32 x) { return (r32)(12*sin(x) - 5 * cos(x)); }

update_result UpdateAndRender(pixel_buffer *PixelBuffer) {
  update_result result = {};

  int width = PixelBuffer->width;
  int height = PixelBuffer->height;

  // Clear screen
  memset(PixelBuffer->memory, 0, width * height * sizeof(u32));

  DrawLine(PixelBuffer, {width/2, 0}, {width/2, height}, 0xFFFFFFFF);
  DrawLine(PixelBuffer, {0, height/2}, {width, height/2}, 0xFFFFFFFF);

  // TODO: draw axis and graphs, then implement the integer bresenham's


  return result;
}
