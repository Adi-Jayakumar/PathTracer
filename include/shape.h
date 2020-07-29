#pragma once
#include "ptutility.h"
#include "ray.h"

class Shape
{
public:
    virtual double Intersect(Ray &ray) = 0;
    virtual Vec Normal(Vec &x) = 0;
    virtual void Translate(Vec &x) = 0;
};