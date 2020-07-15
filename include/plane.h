#pragma once
#include "solid.h"

class Plane : public Solid
{
    public:
        Vec n; // normal
        Plane(Vec _n, Vec _p, Vec _e, Vec _c, Surface _s);
        Plane(const Plane &plane);
        double Intersect(Ray &ray);
        Vec Normal(Vec &x);
        void Translate(Vec &x);
};