#pragma once
#include "vector.h"



namespace PTMath
{
    // image settings
    const int W = 128;
    const int H = 128;
    const int NumSamps = 400;
    const int SubPixSize = 2;

    // geometric constant pi
    const double PI = 3.1415926535897932384626433832795;
    
    // error tolerance for ray Solid intersections
    const double EPSILON = 1e-7;

    // maximum recursion depth
    const int MaxDepth = 10;
    
    // useful functions
    double Luma(const Vec &colour);
    double Random();

}