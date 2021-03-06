#ifndef GRAPHS_CORE_H
#define GRAPHS_CORE_H

#include "base.h"
#include "graphs_math.h"

struct v2i {
  int x;
  int y;
};

inline v2i operator-(v2i A, v2i B) {
  v2i result = {A.x - B.x, A.y - B.y};
  return result;
}

inline v2i operator+(v2i A, v2i B) {
  v2i result = {A.x + B.x, A.y + B.y};
  return result;
}


struct pixel_buffer {
  int width;
  int height;
  int max_width;
  int max_height;
  void *memory;
};

struct update_result {
  // empty for now
};

struct board_state {
  v2 origin;
  m3x3 transform_matrix;
};

struct user_input {

};

update_result UpdateAndRender(pixel_buffer *PixelBuffer, board_state *State,
                              user_input Input);
void AdjustShiftComponent(m3x3 *Matrix, v2 delta);
r32 GetScaleFactor(m3x3 Matrix);
void SetScaleFactor(m3x3 *Matrix, r32 S);
v2 Transform(m3x3 Matrix, v2 Vector);

#endif  // GRAPHS_CORE_H