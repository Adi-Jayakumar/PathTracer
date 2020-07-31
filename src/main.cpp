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

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{

    Scene scene;
    Camera CamUp = Camera(20, PTUtility::W, PTUtility::H, Vec(0, 0, -19.99), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    scene.AddCamera(CamUp);

    scene.LoadCornell(20);

    std::shared_ptr<Sphere> left = std::make_shared<Sphere>(8., Vec(-10, -12, 10));
    std::shared_ptr<Sphere> right = std::make_shared<Sphere>(8., Vec(10, -12, 0));

    Solid mirr = Solid(left, Vec(), Vec(1, 1, 1), Surface::SPEC);
    Solid glass = Solid(right, Vec(), Vec(1, 1, 1), Surface::REFR);

    scene.AddSolid(mirr);
    scene.AddSolid(glass);


    auto start = std::chrono::high_resolution_clock::now();

    scene.TakePicture(0);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << std::endl << "Seconds: " << duration.count() << std::endl;

    return 0;
}