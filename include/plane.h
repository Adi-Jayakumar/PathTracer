#pragma once
#include "solid.h"

class Plane : public Solid
{
    public:
        Vec n; // normal
        Plane(Vec _n, Vec _p, Vec _e, Vec _c, Surface _s);
        double Intersect(Ray &ray, double tMin);
        Vec Normal(Vec &x);
        void Translate(Vec &x);
};