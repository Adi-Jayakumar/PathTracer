#pragma once
#include <vector>
#include "sphere.h"
#include "hitrecord.h"

struct Scene
{
    std::vector<Sphere> objects;
    void AddSphere(Sphere s);
    HitRecord ClosestIntersection(Ray r, double tMin);
};