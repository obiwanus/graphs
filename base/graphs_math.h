#ifndef GRAPHS_MATH_H
#define GRAPHS_MATH_H

#include "base.h"

union v2 {
  struct {
    r32 x, y;
  };
  struct {
    r32 u, v;
  };
  r32 E[2];
};

union v3 {
  struct {
    r32 x, y, z;
  };
  struct {
    r32 u, v, w;
  };
  struct {
    r32 r, g, b;
  };
  struct {
    v2 xy;
    r32 _ignored0;
  };
  struct {
    r32 _ignored1;
    v2 yz;
  };
  struct {
    v2 uv;
    r32 _ignored2;
  };
  struct {
    r32 _ignored3;
    v2 vw;
  };
  r32 E[3];
};

// TODO:

// (1 0 0)
// (0 1 0)
// (0 0 1)
// Find a way to compute graph values out of pixels
// using that matrix (not necessarily nice)

// * Add a matrix struct and operations
// * Take 1 as unit width


#endif  // GRAPHS_MATH_H