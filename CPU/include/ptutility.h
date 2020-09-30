#pragma once
#include "vector.h"
#include <cmath>
#include <random>

namespace PTUtility
{
    // image parameters
    const long int W = 100;
    const long int H = 100;
    const long int NumSamps = 200;
    const long int SubPixSize = 2;
    const int MaxDepth = 10;

    // geometric constant pi
    const double PI = 3.1415926535897932384626433832795;

    // error tolerance for Ray-Shape intersections
    const double EPSILON = 1e-7;

    /*
    maps an rgb colour to a grayscale value betweeon 0 and 1
    to determine Russian Roulette probability for termination
    - an empirical formula
    */
    double Luma(const Vec &colour);

    // non deterministic RNG between 0 and 1
    double Random();

    /*
    solves the qudratic At^2 + Bt + C = 0 if no solutions
    returns double's max value
    */
    std::pair<double, double> SolveQuadratic(double A, double B, double C);
}