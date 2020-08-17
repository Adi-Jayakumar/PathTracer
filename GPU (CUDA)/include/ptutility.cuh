#pragma once
#include <curand_kernel.h>
#include "pair.cuh"
#include "vector.cuh"

namespace PTUtility
{
    // image parameters
    const long int W = 300;
    const long int H = 300;
    const long int NumSamps = 100;
    const long int SubPixSize = 2;

    // geometric constant pi
    const double PI = 3.1415926535897932384626433832795;

    // error tolerance for Ray-Shape intersections
    const double EPSILON = 1e-7;
    const double INF = 1e300;

    // maximum recursion depth
    const int MaxDepth = 3;

    /* maps an rgb colour to a grayscale value betweeon 0 and 1
    to determine Russian Roulette probability for termination
    based off an empirical formula */
    __device__  double Luma(const Vec &colour);

    // non deterministic RNG between 0 and 1
    __device__  double Random(curandState &state);

    /* solves the qudratic At^2 + Bt + C = 0 and checks solutions
    are within tolerances and are positive, if no solutions
    returns double's max value */
    __device__  Pair SolveQuadratic(double A, double B, double C);
    
    __device__ double Max(double a, double b);
}