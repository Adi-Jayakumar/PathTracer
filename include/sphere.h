#pragma once
#include "vector.h"
#include "solid.h"
#include "ray.h"
#include "ptmath.h"

class Sphere : public Solid
{
    public:
        double r;    // radius
        // Vec p, e, c; // position, emmission, colour
        // Surface s;   //surface type
        Sphere(double _r, Vec _p, Vec _e, Vec _c, Surface _s);
        double Intersect(Ray &ray, double tMin);
        Vec Normal(Vec &v);
};