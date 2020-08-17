#include <iostream>
#include "ray.cuh"

__device__ Ray::Ray()
{
    o = Vec();
    d = Vec();
}

__device__ Ray::Ray(Vec _o, Vec _d)
{
    o = _o;
    d = _d;
}

std::ostream &operator<<(std::ostream &out, Ray const &r)
{
    out << "Ray origin: " << r.o << " "<< "Ray dir: " << r.d << std::endl;
    return out;
}