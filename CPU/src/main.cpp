#include "camera.h"
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

// RUN: clear && make && ./bin/main
int main()
{

    Scene scene;

    scene.LoadCornell(10);
    std::shared_ptr<Sphere> mirror = std::make_shared<Sphere>(5, Vec(-5, -5, 2));
    std::shared_ptr<Sphere> glass = std::make_shared<Sphere>(5, Vec(5, -5, -5));
    // scene.AddSolid(Solid(mirror, Vec(), Vec(1, 1, 1), Surface::SPEC));
    // scene.AddSolid(Solid(glass, Vec(), Vec(1, 1, 1), Surface::REFRGLOSS));

    // clock start
    auto start = std::chrono::high_resolution_clock::now();

    scene.TakePicture(0);

    // clock stop
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << std::endl
              << "Seconds: " << duration.count() << std::endl;

    return 0;
}