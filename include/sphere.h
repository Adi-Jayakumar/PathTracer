#pragma once
#pragma once
#include "vector.h"
#include "shape.h"
#include "ray.h"

struct Sphere : Shape
{
    public:
        double r;
        Vec c;
        Sphere(double _r, Vec _c);
        bool Intersect(Ray &ray, double & hit) override;
        Vec Normal(Vec &v) override;
        void Translate(Vec &x) override;
        bool IsOnSkin(Vec &x) override;
        double FarSolution(Ray &ray) override;
};