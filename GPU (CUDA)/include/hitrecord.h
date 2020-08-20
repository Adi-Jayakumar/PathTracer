#pragma once
#include <iostream>

struct HitRecord
{
    float t; // value of the line parameter at the hitpoint --> inf if no hit
    int id;  // index of the sphere that was hit --> -1 if not hit
    __device__ HitRecord(float _t, int _id);
    __device__ HitRecord()
    {
        t = 0;
        id = -1;
    };
    friend std::ostream &operator<<(std::ostream &out, HitRecord const &h);
};