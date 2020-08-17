#include <iostream>
#include <cmath>
#include "vector.cuh"

__host__ __device__ Vec::Vec()
{
    x = 0;
    y = 0;
    z = 0;
}
__host__ __device__ Vec::Vec(double _x, double _y, double _z)
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

__host__ __device__ Vec Vec::operator*(double k)
{
    return Vec(x * k, y * k, z * k);
}
__host__ __device__ Vec Vec::operator/(double k)
{
    return Vec(x / k, y / k, z / k);
}
__host__ __device__ Vec Vec::Cross(Vec u, Vec v)
{
    return Vec(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
__host__ __device__ double Vec::Mod()
{
    return sqrt(x * x + y * y + z * z);
}
__host__ __device__ double Vec::ModSq()
{
    return x * x + y * y + z * z;
}
__host__ __device__ Vec Vec::Norm()
{
    double size = sqrt(x * x + y * y + z * z);
    return Vec(x / size, y / size, z / size);
}
__host__ __device__ double Vec::Dot(Vec u, Vec v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}