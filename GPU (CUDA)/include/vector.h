#pragma once
#include <ostream>
struct Vec
{
    double x, y, z; // x, y, z components
    __host__ __device__ Vec();
    __host__ __device__ Vec(double _x, double _y, double _z);
    friend std::ostream &operator<<(std::ostream &out, Vec const &v);
    __host__ __device__ friend Vec operator+(Vec u, Vec v);
    __host__ __device__ friend Vec operator-(Vec u, Vec v);
    __host__ __device__ friend Vec operator*(Vec u, Vec v);
    __host__ __device__ Vec operator*(double k);
    __host__ __device__ Vec operator/(double k);
    __host__ __device__ static Vec Cross(Vec u, Vec v); // cross product of 2 vectors
    __host__ __device__ double Mod(); // returns the modulus of the vector
    __host__ __device__ double ModSq(); // returns the modulus of the vector squared
    __host__ __device__ static double Dot(Vec u, Vec v); // dot product of 2 vectors
    __host__ __device__ Vec Norm(); // returns a normalised vector
};
