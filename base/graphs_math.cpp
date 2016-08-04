#include <math.h>
#include "graphs_math.h"

// ==================== Construction ======================

inline v2
V2i(i32 X, i32 Y)
{
    v2 Result = {(r32)X, (r32)Y};

    return(Result);
}

inline v2
V2i(u32 X, u32 Y)
{
    v2 Result = {(r32)X, (r32)Y};

    return(Result);
}

inline v2
V2(r32 X, r32 Y)
{
    v2 Result;

    Result.x = X;
    Result.y = Y;

    return(Result);
}

inline v3
V3(r32 X, r32 Y, r32 Z)
{
    v3 Result;

    Result.x = X;
    Result.y = Y;
    Result.z = Z;

    return(Result);
}

inline v3
V3(v2 XY, r32 Z)
{
    v3 Result;

    Result.x = XY.x;
    Result.y = XY.y;
    Result.z = Z;

    return(Result);
}

// ======================= Scalar operations =======================

inline r32 Square(r32 A) {
  r32 Result = A * A;

  return Result;
}

inline r32 Lerp(r32 A, r32 t, r32 B) {
  r32 Result = (1.0f - t) * A + t * B;

  return Result;
}

inline r32 Clamp(r32 Min, r32 Value, r32 Max) {
  r32 Result = Value;

  if (Result < Min) {
    Result = Min;
  } else if (Result > Max) {
    Result = Max;
  }

  return Result;
}

inline r32 Clamp01(r32 Value) {
  r32 Result = Clamp(0.0f, Value, 1.0f);

  return Result;
}

inline r32 Clamp01MapToRange(r32 Min, r32 t, r32 Max) {
  r32 Result = 0.0f;

  r32 Range = Max - Min;
  if (Range != 0.0f) {
    Result = Clamp01((t - Min) / Range);
  }

  return Result;
}

inline r32 SafeRatioN(r32 Numerator, r32 Divisor, r32 N) {
  r32 Result = N;

  if (Divisor != 0.0f) {
    Result = Numerator / Divisor;
  }

  return Result;
}

inline r32 SafeRatio0(r32 Numerator, r32 Divisor) {
  r32 Result = SafeRatioN(Numerator, Divisor, 0.0f);

  return Result;
}

inline r32 SafeRatio1(r32 Numerator, r32 Divisor) {
  r32 Result = SafeRatioN(Numerator, Divisor, 1.0f);

  return Result;
}

// ========================== v2 operations ========================

inline v2 Perp(v2 A) {
  v2 Result = {-A.y, A.x};
  return Result;
}

inline v2 operator*(r32 A, v2 B) {
  v2 Result;

  Result.x = A * B.x;
  Result.y = A * B.y;

  return Result;
}

inline v2 operator*(v2 B, r32 A) {
  v2 Result = A * B;

  return Result;
}

inline v2 &operator*=(v2 &B, r32 A) {
  B = A * B;

  return B;
}

inline v2 operator-(v2 A) {
  v2 Result;

  Result.x = -A.x;
  Result.y = -A.y;

  return Result;
}

inline v2 operator+(v2 A, v2 B) {
  v2 Result;

  Result.x = A.x + B.x;
  Result.y = A.y + B.y;

  return Result;
}

inline v2 &operator+=(v2 &A, v2 B) {
  A = A + B;

  return A;
}

inline v2 operator-(v2 A, v2 B) {
  v2 Result;

  Result.x = A.x - B.x;
  Result.y = A.y - B.y;

  return Result;
}

inline v2 Hadamard(v2 A, v2 B) {
  v2 Result = {A.x * B.x, A.y * B.y};

  return Result;
}

inline r32 operator*(v2 A, v2 B) {
  r32 Result = A.x * B.x + A.y * B.y;

  return Result;
}

inline r32 LengthSq(v2 A) {
  r32 Result = A * A;

  return Result;
}

inline r32 Length(v2 A) {
  r32 Result = sqrt(LengthSq(A));
  return Result;
}

// ============================= v3 operations =======================

inline v3 operator*(r32 A, v3 B) {
  v3 Result;

  Result.x = A * B.x;
  Result.y = A * B.y;
  Result.z = A * B.z;

  return Result;
}

inline v3 operator*(v3 B, r32 A) {
  v3 Result = A * B;

  return Result;
}

inline v3 &operator*=(v3 &B, r32 A) {
  B = A * B;

  return B;
}

inline v3 operator-(v3 A) {
  v3 Result;

  Result.x = -A.x;
  Result.y = -A.y;
  Result.z = -A.z;

  return Result;
}

inline v3 operator+(v3 A, v3 B) {
  v3 Result;

  Result.x = A.x + B.x;
  Result.y = A.y + B.y;
  Result.z = A.z + B.z;

  return Result;
}

inline v3 &operator+=(v3 &A, v3 B) {
  A = A + B;

  return A;
}

inline v3 operator-(v3 A, v3 B) {
  v3 Result;

  Result.x = A.x - B.x;
  Result.y = A.y - B.y;
  Result.z = A.z - B.z;

  return Result;
}

inline v3 Hadamard(v3 A, v3 B) {
  v3 Result = {A.x * B.x, A.y * B.y, A.z * B.z};

  return Result;
}

inline r32 operator*(v3 A, v3 B) {
  r32 Result = A.x * B.x + A.y * B.y + A.z * B.z;

  return Result;
}

inline r32 LengthSq(v3 A) {
  r32 Result = A * A;

  return Result;
}

inline r32 Length(v3 A) {
  r32 Result = sqrt(LengthSq(A));
  return Result;
}

inline v3 Normalize(v3 A) {
  v3 Result = A * (1.0f / Length(A));

  return Result;
}

