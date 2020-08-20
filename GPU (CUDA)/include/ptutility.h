#pragma once
#include "pair.h"
#include "vector.h"
#include <curand_kernel.h>

namespace PTUtility
{
    // image parameters
    const long int W = 1024;
    const long int H = 1024;
    const long int NumSamps = 1500;
    const long int SubPixSize = 2;

    // geometric constant pi
    const float PI = 3.1415926535897932384626433832795;

    // error tolerance for Ray-Shape intersections
    const float EPSILON = 1e-7;
    const float INF = 1e300;

    // maximum recursion depth
    const int MaxDepth = 12;

    /* maps an rgb colour to a grayscale value betweeon 0 and 1
    to determine Russian Roulette probability for termination
    based off an empirical formula */
    __device__ float Luma(const Vec &colour);

    // non deterministic RNG between 0 and 1
    __device__ float Random(curandState &state);

    /* solves the qudratic At^2 + Bt + C = 0 and checks solutions
    are within tolerances and are positive, if no solutions
    returns float's max value */
    __device__ Pair SolveQuadratic(float A, float B, float C);

    __device__ float Max(float a, float b);
} // namespace PTUtility