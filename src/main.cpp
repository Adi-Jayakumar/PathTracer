#include "camera.h"
#include "composite.h"
#include "hitrecord.h"
#include "image.h"
#include "plane.h"
#include "ptutility.h"
#include "scene.h"
#include "sphere.h"
#include "triangle.h"
#include "vector.h"
#include <chrono>
#include <cmath>
#include <iostream>

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{

    Scene scene;
    Camera CamUp = Camera(1, PTUtility::W, PTUtility::H, Vec(0, 0, -1.99), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    scene.AddCamera(CamUp);

    scene.LoadCornell(2);

    std::shared_ptr<Sphere> right = std::make_shared<Sphere>(1, Vec(0.5, 0, 0));
    std::shared_ptr<Sphere> left = std::make_shared<Sphere>(1, Vec(-0.5, 0, 0));
    std::shared_ptr<Composite> comp = right - left;

    auto start = std::chrono::high_resolution_clock::now();

    scene.TakePicture(0);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << std::endl
              << "Seconds: " << duration.count() << std::endl;

    return 0;
}