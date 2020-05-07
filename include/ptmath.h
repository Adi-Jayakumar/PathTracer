#pragma once
#include "vector.h"

enum Surface
{
    DIFF,
    SPEC,
    REFR
};

namespace PTMath
{
    const double PI = 3.1415926535897932384626433832795;
    const int MaxDepth = 10;
    double Luma(const Vec &colour);
    double Random();
    double Tent();

}