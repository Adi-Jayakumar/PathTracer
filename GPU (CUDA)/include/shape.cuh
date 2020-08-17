#pragma once
#include "ptutility.cuh"
#include "ray.cuh"
#include <memory>
#include <vector>

struct Shape
{
    // thses 2 are needed for path tracing
    __device__ virtual bool Intersect(Ray &ray, double &hit) = 0;
    __device__ virtual Vec Normal(Vec &x) = 0;
    __device__ virtual void Translate(Vec &x) = 0;
};