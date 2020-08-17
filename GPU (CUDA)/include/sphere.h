#pragma once
#include "vector.h"
#include "shape.h"
#include "ray.h"


struct Sphere : Shape
{
    double r;
    Vec c;
    __device__ Sphere(double _r, Vec _c);
    __device__ bool Intersect(Ray &ray, double &hit) override;
    __device__ Vec Normal(Vec &v) override;
    __device__ void Translate(Vec &x) override;
};