#include "solid.cuh"

__device__ __host__ Solid::Solid()
{
    shape = nullptr;
    e = Vec();
    c = Vec();
    s = Surface::DIFF;
}

__device__ __host__ Solid::Solid(Shape*_shape, Vec _e, Vec _c, Surface _s)
{
    shape = _shape;
    e = _e;
    c = _c;
    s = _s;
}