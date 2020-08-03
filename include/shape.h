#pragma once
#include "ptutility.h"
#include "ray.h"

class Shape
{
public:
    virtual bool Intersect(Ray &ray, double &hit) = 0;
    virtual Vec Normal(Vec &x) = 0;
    virtual void Translate(Vec &x) = 0;
    virtual bool IsOnSkin(Vec &x) = 0;
};