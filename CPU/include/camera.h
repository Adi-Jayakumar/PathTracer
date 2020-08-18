#pragma once
#include "vector.h"
#include "ray.h"
#include "ptutility.h"
#include <cmath>

struct Camera
{
    Vec loc, forward, up, right, focus; // loc is the centre point of the screen
    double worldW, worldH, hFOV; // width and height of view finder in world, horixontal fov
    int nPixelsX, nPixelsY; // number of pixels in x and y directions
    Camera(double _worldW, int _nPixelsX, int _nPixelsY, Vec _loc, Vec _forward, Vec _up, double _hFOV);
    Ray GenerateRay(int i, int j, int sx, int sy);
};