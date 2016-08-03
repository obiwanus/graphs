#ifndef GRAPHS_CORE_H
#define GRAPHS_CORE_H

#include "base.h"

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

struct v3f {
  r32 x;
  r32 y;
  r32 z;
};

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
  int unit_width;
  v2i origin;
};

update_result UpdateAndRender(pixel_buffer *PixelBuffer, board_state *State);

#endif  // GRAPHS_CORE_H