#pragma once
#include "shape.h"
#include "vector.h"
#include "ray.h"

class Triangle : public Shape
{ 
    public:
        Vec p1, p2, p3, n;
        Triangle(Vec _p1, Vec _p2, Vec _p3);
        double Intersect(Ray &r);
        Vec Normal(Vec &x);
        void Translate(Vec &x);
};