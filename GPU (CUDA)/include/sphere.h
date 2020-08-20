#pragma once
#include "vector.h"
#include "shape.h"
#include "ray.h"


struct Sphere : Shape
{
    float r;
    Vec c;
    __device__ Sphere(float _r, Vec _c);
    __device__ bool Intersect(Ray &ray, float &hit) override;
    __device__ Vec Normal(Vec &v) override;
    __device__ void Translate(Vec &x) override;
};