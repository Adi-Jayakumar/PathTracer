#pragma once
#include "vector.h"
#include "ptmath.h"
#include "ray.h"

class Solid
{   
    public:
        Vec p, e, c; 
        Surface s;
        virtual double Intersect(Ray &ray, double tMin) = 0;
        virtual Vec Normal(Vec &x) = 0;
};