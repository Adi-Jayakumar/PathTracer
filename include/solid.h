#pragma once
#include "ray.h"
#include "ptutility.h"

enum class Surface
{
    DIFF,
    SPEC,
    REFR
};

class Solid
{
public:
    Vec p, e, c;
    Surface s;
    virtual double Intersect(Ray &ray) = 0;
    virtual Vec Normal(Vec &x) = 0;
    virtual void Translate(Vec &x) = 0;
};