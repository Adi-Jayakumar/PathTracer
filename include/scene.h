#pragma once
#include <vector>
#include "solid.h"
#include "sphere.h"
#include "hitrecord.h"

struct Scene
{
    std::vector<Solid*> objects;
    void AddSphere(Solid* s);
    HitRecord ClosestIntersection(Ray r, double tMin);
};