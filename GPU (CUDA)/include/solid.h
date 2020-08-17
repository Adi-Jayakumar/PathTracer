#pragma once
#include "shape.h"
#include "ptutility.h"

enum class Surface
{
    DIFF,
    SPEC,
    SPECGLOSS,
    REFRGLOSS,
    REFR,
};

struct Solid
{
    Shape* shape;
    Vec e, c;
    Surface s;
    __device__ __host__ Solid();
    __device__ __host__ Solid(Shape* _shape, Vec _e, Vec _c, Surface _s);
    __device__ __host__ ~Solid(){delete shape;}
};