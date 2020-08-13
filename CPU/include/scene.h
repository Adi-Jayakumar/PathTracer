#pragma once
#include "camera.h"
#include "hitrecord.h"
#include "solid.h"
#include "sphere.h"
#include "triangle.h"

struct Scene
{
    std::vector<Solid> objects;
    std::vector<Camera> cameras;
    void AddSolid(Solid s);
    void AddCamera(Camera c);
    HitRecord ClosestIntersection(Ray r);
    Vec RayColour(Ray r, int depth);
    void TakePicture(int index);
    void LoadOBJModel(std::string fPath);
    void LoadCornell(double boxSize);
};