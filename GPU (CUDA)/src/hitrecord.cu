#include <iostream>
#include "hitrecord.cuh"

__device__ HitRecord::HitRecord(double _t, int _id)
{   
    t = _t;
    id = _id;
}

std::ostream &operator<<(std::ostream &out, HitRecord const &h)
{
    out << "Hit t = " << h.t << " " << "Hit id = " << h.id << std::endl;
    return out;
}