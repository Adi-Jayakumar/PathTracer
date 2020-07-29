#pragma once
#pragma once
#include "vector.h"
#include "shape.h"
#include "ray.h"

class Sphere : public Shape
{
    public:
        double r;
        Vec c;
        Sphere(double _r, Vec _c);
        double Intersect(Ray &ray);
        Vec Normal(Vec &v);
        void Translate(Vec &x);
};