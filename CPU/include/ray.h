#pragma once
#include "vector.h"
#include <iostream>
struct Ray
{   
    Vec o, d; // origin, direction
    Ray();
    Ray(Vec _o, Vec _d);
    friend std::ostream &operator<<(std::ostream &out, Ray const &h);
};