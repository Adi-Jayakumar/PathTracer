#pragma once
#include "camera.cuh"
#include "hitrecord.cuh"
#include "solid.cuh"

namespace Scene
{
    __device__ HitRecord ClosestIntersection(Ray r, Solid** objects, int nObj);
    __device__ Ray GenerateDiffuseRay(Vec& hitPt, Vec& normal, curandState &state);
    __device__ Ray GenerateSpecularRay(Vec& hitPt, Vec& normal, Vec &dir);
    __device__ Vec Jitter(Vec x, double r, curandState &state);
    __device__ Vec RayColour(Ray r,Solid** objects, curandState& state, int nObj);
}