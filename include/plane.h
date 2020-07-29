#pragma once
#include "solid.h"

class Plane : public Shape
{
    public:
        Vec n; // normal
        Vec p;
        Plane(Vec _n, Vec _p);
        Plane(const Plane &plane);
        double Intersect(Ray &ray);
        Vec Normal(Vec &x);
        void Translate(Vec &x);
};