#pragma once
#include "image.h"
#include "ray.h"
#include <memory>

struct Shape
{
    // thses 2 are needed for path tracing
    virtual bool Intersect(Ray &ray, double &hit) = 0;
    virtual Vec Normal(Vec &x) = 0;
    virtual void Translate(Vec &x) = 0;
};