#pragma once
#include "image.h"
#include "ray.h"
#include <memory>

struct Shape
{
    // thses 2 are needed for path tracing
    virtual bool Intersect(Ray &ray, double &hit, std::shared_ptr<std::pair<double, double>> values = nullptr) = 0;
    virtual Vec Normal(Vec &x) = 0;

    virtual void Translate(Vec &x) = 0;

    // these 2 are used in Composite
    virtual bool IsOnSkin(Vec &x) = 0;
};