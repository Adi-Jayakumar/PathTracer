#pragma once
#include "vector.h"

namespace PTUtility
{
    // image parameters
    const int W = 100;
    const int H = 100;
    const int NumSamps = 400;
    const int SubPixSize = 2;

    // geometric constant pi
    const double PI = 3.1415926535897932384626433832795;

    // error tolerance for Ray-Shape intersections
    const double EPSILON = 1e-7;

    // maximum recursion depth
    const int MaxDepth = 10;

    /* maps an rgb colour to a grayscale value betweeon 0 and 1
    to determine Russian Roulette probability for termination
    based off an empirical formula */
    double Luma(const Vec &colour);

    // non deterministic RNG between 0 and 1
    double Random();

    /* solves the qudratic At^2 + Bt + C = 0 and checks solutions 
    are within tolerances and are positive, if no solutions 
    returns double's max value */
    double SolveQuadratic(double A, double B, double C);
}