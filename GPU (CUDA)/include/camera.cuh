#pragma once
#include "vector.cuh"
#include "ray.cuh"
#include "ptutility.cuh"
#include <curand_kernel.h>

struct Camera
{
    Vec loc, forward, up, right, focus; // loc is the centre point of the screen
    double worldW, worldH, hFOV; // width and height of view finder in world, horixontal fov
    int nPixelsX, nPixelsY; // number of pixels in x and y directions
    __device__ Camera();
    __device__ Camera(double _worldW, int _nPixelsX, int _nPixelsY, Vec _loc, Vec _forward, Vec _up, double _hFOV);
    __device__ Ray GenerateAARay(int i, int j, int sx, int sy, curandState &state);
    __device__ Ray GenerateRay(int i, int j);
};