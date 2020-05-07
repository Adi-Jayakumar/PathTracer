#include <iostream>
#include <cmath>
#include "vector.h"

Vec::Vec()
{
    x = 0;
    y = 0;
    z = 0;
}
Vec::Vec(double _x, double _y, double _z)
{
    x = _x;
    y = _y;
    z = _z;
}

std::ostream &operator<<(std::ostream &out, Vec const &v)
{
    // out << "(" << v.x << ", " << v.y << ", " << v.z << ")";// << std::endl;
    out << v.x << ", " << v.y << ", " << v.z << "   "; // << std::endl;
    return out;
}

Vec operator+(Vec u, Vec v)
{
    return Vec(u.x + v.x, u.y + v.y, u.z + v.z);
}

Vec operator-(Vec u, Vec v)
{
    return Vec(u.x - v.x, u.y - v.y, u.z - v.z);
}

Vec operator*(Vec u, Vec v)
{
    return Vec(u.x * v.x, u.y * v.y, u.z * v.z);
}

Vec Vec::operator*(double k)
{
    return Vec(x * k, y * k, z * k);
}
Vec Vec::operator/(double k)
{
    return Vec(x / k, y / k, z / k);
}
Vec Vec::Cross(Vec u, Vec v)
{
    return Vec(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
double Vec::Mod()
{
    return sqrt(x * x + y * y + z * z);
}
Vec Vec::Norm()
{
    double size = sqrt(x * x + y * y + z * z);
    return Vec(x / size, y / size, z / size);
}
double Vec::Dot(Vec u, Vec v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}