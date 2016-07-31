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

void DrawLine(pixel_buffer *PixelBuffer, v2i A, v2i B, u32 Color) {
  bool swapped = false;
  if (abs(B.x - A.x) < abs(B.y - A.y)) {
    int tmp = A.x;
    A.x = A.y;
    A.y = tmp;
    tmp = B.x;
    B.x = B.y;
    B.y = tmp;
    swapped = true;
  }
  if (B.x - A.x < 0) {
    v2i tmp = B;
    B = A;
    A = tmp;
  }

  int dy = B.y - A.y;
  int dx = B.x - A.x;
  int error = dy - dx;
  int y = A.y;
  for (int x = A.x; x <= B.x; x++) {
    if (!swapped) {
      DrawPixel(PixelBuffer, x, y, Color);
    } else {
      DrawPixel(PixelBuffer, y, x, Color);
    }
    error += dy;
    if (error > 0) {
      error -= dx;
      y++;
    }
  }
}

// r32 func(r32 x) { return (r32)(12 * sin(x / 2)); }
r32 func(r32 x) { return 0.5f * x * x; }

update_result UpdateAndRender(pixel_buffer *PixelBuffer, board_state *State) {
  update_result result = {};

  int width = PixelBuffer->width;
  int height = PixelBuffer->height;

  // Clear screen
  memset(PixelBuffer->memory, 0, width * height * sizeof(u32));

  DrawLine(PixelBuffer, {State->origin.x, 0}, {State->origin.x, height},
           0xFFFFFFFF);
  DrawLine(PixelBuffer, {0, State->origin.y}, {width, State->origin.y},
           0xFFFFFFFF);

  int unit_width = State->unit_width;

  int y_pixel_prev = 0;
  for (int x_pixel = -1; x_pixel <= width; x_pixel++) {
    r32 x = (r32)(x_pixel - State->origin.x) / unit_width;
    r32 y = func(x);
    int y_pixel = (int)(y * unit_width) + State->origin.y;
    if (x_pixel >= 0 && x_pixel <= width && y_pixel >= 0 && y_pixel <= height) {
      DrawLine(PixelBuffer, {x_pixel, y_pixel}, {x_pixel, y_pixel_prev},
               0xFF55DD99);
    }
    y_pixel_prev = y_pixel;
  }

  return result;
}
