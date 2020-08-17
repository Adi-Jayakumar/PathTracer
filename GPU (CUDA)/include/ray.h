#pragma once
#include "vector.h"
struct Ray
{   
    Vec o, d; // origin, direction
    __device__ Ray();
    __device__ Ray(Vec _o, Vec _d);
    friend std::ostream &operator<<(std::ostream &out, Ray const &h);
};