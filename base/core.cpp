#include <string.h>
#include <math.h>

#include "core.h"
#include "base.h"

inline void DrawPixel(pixel_buffer *PixelBuffer, v2i Point, u32 Color) {
  int x = Point.x;
  int y = Point.y;

  if (x < 0 || x > PixelBuffer->width || y < 0 || y > PixelBuffer->height) {
    return;
  }
  y = PixelBuffer->height - y;  // Origin in bottom-left
  u32 *pixel = (u32 *)PixelBuffer->memory + x + y * PixelBuffer->width;
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
      DrawPixel(PixelBuffer, {x, y}, Color);
    } else {
      DrawPixel(PixelBuffer, {y, x}, Color);
    }
    error += dy;
    if (error > 0) {
      error -= dx;
      y++;
    }
  }
}

static r32 func(r32 x) { return (r32)(12 * sin(x / 2)); }
// r32 func(r32 x) { return 0.5f * x * x; }

inline r32 GetScaleFactor(m3x3 Matrix) {
  return Matrix.rows[0].x;
}

void AdjustScaleFactor(m3x3 *Matrix, r32 Value) {
  Matrix->e[0] += Value;
  Matrix->e[4] += Value;
  if (Matrix->e[0] < 1) {
    Matrix->e[0] = 1;
    Matrix->e[4] = 1;
  }
}

void AdjustShiftComponent(m3x3 *Matrix, v2 delta) {
  Matrix->rows[0].z += delta.x;
  Matrix->rows[1].z += delta.y;
}

v2 Transform(m3x3 Matrix, v2 Vector) {
  // Note the vector is v2
  v2 result = {};
  v3 v = {Vector.x, Vector.y, 1.0f};

  v = Matrix * v;
  result = {v.x, v.y};

  return result;
}

update_result UpdateAndRender(pixel_buffer *PixelBuffer, board_state *State) {
  update_result result = {};

  int width = PixelBuffer->width;
  int height = PixelBuffer->height;

  // Clear screen
  memset(PixelBuffer->memory, 0, width * height * sizeof(u32));

  r32 unit_width = GetScaleFactor(State->transform_matrix);
  v2 origin = Transform(State->transform_matrix, State->origin);

  // Draw axis
  DrawLine(PixelBuffer, {(int)origin.x, 0}, {(int)origin.x, height},
           0xFFFFFFFF);
  DrawLine(PixelBuffer, {0, (int)origin.y}, {width, (int)origin.y},
           0xFFFFFFFF);

  // Draw graph
  int y_pixel_prev = 0;
  for (int x_pixel = -1; x_pixel <= width; x_pixel++) {
    r32 x = (r32)(x_pixel - origin.x) / unit_width;
    r32 y = func(x);
    int y_pixel = (int)(y * unit_width) + (int)origin.y;
    DrawLine(PixelBuffer, {x_pixel, y_pixel}, {x_pixel, y_pixel_prev},
             0xFFFFDD99);
    y_pixel_prev = y_pixel;
  }

  return result;
}
