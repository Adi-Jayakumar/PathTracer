#pragma once
#include <vector>
#include <memory>
#include "solid.h"
#include "sphere.h"
#include "triangle.h"
#include "hitrecord.h"
#include "camera.h"

class Scene
{
    public:
        std::vector<std::shared_ptr<Solid>> objects;
        std::vector<Camera> cameras;
        void AddSolid(std::shared_ptr<Solid> s);
        void AddCamera(Camera c);
        HitRecord ClosestIntersection(Ray r);
        Vec RayColour(Ray r, int depth);
        void TakePicture(int index);
        void LoadOBJModel(std::string fPath);
        void LoadCornell(double boxSize);
};