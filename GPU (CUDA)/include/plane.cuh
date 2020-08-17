#pragma once
#include "shape.cuh"
#include "ptutility.cuh"

struct Plane : Shape
{
    Vec n; // normal
    Vec p;
    __device__ Plane(Vec _n, Vec _p);
    __device__ bool Intersect(Ray &ray, double &hit) override;
    __device__ Vec Normal(Vec &x) override;
    __device__ void Translate(Vec &x) override;
};