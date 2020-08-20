#include "vector.h"
#include <cmath>
#include <iostream>

__host__ __device__ Vec::Vec()
{
    x = 0;
    y = 0;
    z = 0;
}
__host__ __device__ Vec::Vec(float _x, float _y, float _z)
{
    x = _x;
    y = _y;
    z = _z;
}

std::ostream &operator<<(std::ostream &out, Vec const &v)
{
    // out << "(" << v.x << ", " << v.y << ", " << v.z << ")";// << std::endl;
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")"; // << std::endl;
    return out;
}

__host__ __device__ Vec operator+(Vec u, Vec v)
{
    return Vec(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ Vec operator-(Vec u, Vec v)
{
    return Vec(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ Vec operator*(Vec u, Vec v)
{
    return Vec(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ Vec Vec::operator*(float k)
{
    return Vec(x * k, y * k, z * k);
}
__host__ __device__ Vec Vec::operator/(float k)
{
    return Vec(x / k, y / k, z / k);
}
__host__ __device__ Vec Vec::Cross(Vec u, Vec v)
{
    return Vec(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
__host__ __device__ float Vec::Mod()
{
    return sqrt(x * x + y * y + z * z);
}
__host__ __device__ float Vec::ModSq()
{
    return x * x + y * y + z * z;
}
__host__ __device__ Vec Vec::Norm()
{
    float size = sqrt(x * x + y * y + z * z);
    return Vec(x / size, y / size, z / size);
}
__host__ __device__ float Vec::Dot(Vec u, Vec v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}