#pragma once
#include "camera.h"
#include "hitrecord.h"
#include "solid.h"
#include "pair.h"

namespace Scene
{
    __device__ HitRecord ClosestIntersection(Ray r, Solid** objects, int nObj);
    __device__ Ray GenerateDiffuseRay(Vec& hitPt, Vec& normal, curandState &state);
    __device__ Ray GenerateSpecularRay(Vec& hitPt, Vec& normal, Vec &dir);
    __device__ Vec Jitter(Vec x, float r, curandState &state);
    __device__ Ray GenerateRefractedRay(Vec &rayNormal, Vec &normal, Vec &hitPt, Vec &dir);
    __device__ Pair GetRefractionWeights(Vec &rayNormal, Vec &normal, Vec &refractedDir, Vec &dir);
    __device__ Vec RayColour(Ray r,Solid** objects, curandState& state, int nObj);
}