#include <iostream>
#include "ray.h"

Ray::Ray()
{
    o = Vec();
    d = Vec();
}

Ray::Ray(Vec _o, Vec _d)
{
    o = _o;
    d = _d;
}

std::ostream &operator<<(std::ostream &out, Ray const &r)
{
    out << "Ray origin: " << r.o << " "<< "Ray dir: " << r.d << std::endl;
    return out;
}