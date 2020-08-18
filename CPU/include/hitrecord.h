#pragma once
#include <iostream>

struct HitRecord
{
    double t; // value of the line parameter at the hitpoint --> inf if no hit
    int id; // index of the sphere that was hit --> -1 if not hit
    HitRecord(double _t, int _id);
    friend std::ostream &operator <<(std::ostream &out, HitRecord const &h);
};