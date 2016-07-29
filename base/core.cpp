#include <string.h>
#include <math.h>

#include "core.h"
#include "base.h"

// NOTE: a temporary function
inline void DrawPixel(pixel_buffer *PixelBuffer, int X, int Y, u32 Color) {
  if (X < 0 || X > PixelBuffer->width || Y < 0 || Y > PixelBuffer->height) {
    return;
  }
  Y = PixelBuffer->height - Y;
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

// NOTE: a temporary function
void DrawLine(pixel_buffer *PixelBuffer, v2i A, v2i B, u32 Color) {
  v2i left = A;
  v2i right = B;

  if (A.x == B.x) {
    int step = (A.y < B.y) ? 1 : -1;
    int y = A.y;
    while (y != B.y) {
      DrawPixel(PixelBuffer, A.x, y, Color);
      y += step;
    }
    return;
  }

  if (A.y == B.y) {
    int step = (A.x < B.x) ? 1 : -1;
    int x = A.x;
    while (x != B.x) {
      DrawPixel(PixelBuffer, x, A.y, Color);
      x += step;
    }
    return;
  }

  if (A.x > B.x) {
    left = B;
    right = A;
  }

  for (int x = left.x; x < right.x; x++) {
    r32 t = (x - left.x) / (r32)(right.x - left.x);
    int y = (int)(left.y * (1.0f - t) + right.y * t);
    DrawPixel(PixelBuffer, x, y, Color);
  }
}

r32 func(r32 x) { return (r32)(12*sin(x) - 5 * cos(x)); }

update_result UpdateAndRender(pixel_buffer *PixelBuffer) {
  update_result result = {};

  // Clear screen
  memset(PixelBuffer->memory, 0,
         PixelBuffer->width * PixelBuffer->height * sizeof(u32));

  int width = PixelBuffer->width;
  int height = PixelBuffer->height;

  // Draw the axis
  DrawLine(PixelBuffer, {width / 2, 0}, {width / 2, height}, 0xFFFFFFFF);
  DrawLine(PixelBuffer, {0, height / 2}, {width, height / 2}, 0xFFFFFFFF);

  int unit_width = 20;

  int min_x = (-width / 2) / unit_width;
  int max_x = (width / 2) / unit_width;
  for (int x = min_x; x <= max_x; x++) {
    int actual_x = (x * unit_width) + width / 2;
    DrawLine(PixelBuffer, {actual_x, (height / 2) - 1},
             {actual_x, (height / 2) + 2}, 0xFFFFFFFF);
  }

  int min_y = (-height / 2) / unit_width;
  int max_y = (height / 2) / unit_width;
  for (int y = min_y; y <= max_y; y++) {
    int actual_y = (y * unit_width) + height / 2;
    DrawLine(PixelBuffer, {(width / 2) - 1, actual_y},
             {(width / 2) + 2, actual_y}, 0xFFFFFFFF);
  }

  // Draw the graph
  int prev_y_pixel = -1;
  for (int x_pixel = 0; x_pixel <= width; x_pixel++) {
    r32 x = (r32)min_x + (r32)x_pixel / (r32)unit_width;
    r32 y = func(x);
    int y_pixel = (int)(y * unit_width) + height / 2;
    if (x_pixel > 0) {
      DrawLine(PixelBuffer, {x_pixel, prev_y_pixel}, {x_pixel, y_pixel}, 0xFF33FFAA);
    } else {
      DrawPixel(PixelBuffer, x_pixel, y_pixel, 0xFF33FFAA);
    }
    prev_y_pixel = y_pixel;
  }

  return result;
}
