#pragma once
#include <vector>
#include "solid.h"
#include "sphere.h"
#include "hitrecord.h"
#include "camera.h"

struct Scene
{
    std::vector<Solid*> objects;
    std::vector<Camera> cameras;
    void AddSolid(Solid* s);
    void AddCamera(Camera c);
    HitRecord ClosestIntersection(Ray r);
    Vec PixelColour(Ray r, int depth);
    void TakePicture(int index);
};