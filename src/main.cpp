#include <iostream>
#include <cmath>
#include "vector.h"
#include "scene.h"
#include "sphere.h"
#include "triangle.h"
#include "plane.h"
#include "image.h"
#include "hitrecord.h"
#include "camera.h"
#include "ptutility.h"

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{

    Scene scene;
    // double boxSize = 15;
    // Camera CamUp = Camera(boxSize, PTUtility::W, PTUtility::H, Vec(0, 0, -boxSize + 0.1), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    Camera CamUp = Camera(2, PTUtility::W, PTUtility::H, Vec(0, 0, -1.99), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    scene.AddCamera(CamUp);

    // std::shared_ptr<Solid> mirr = std::make_shared<Triangle>(Vec(0, 2, 1.99), Vec(-2, -2, 1.99), Vec(2, -2, 1.99), Vec(0, 0, 0), Vec(1, 1, 1), SPEC);
    // Vec x = Vec(0, 0, 0);
    // cout << mirr->Normal(x) << endl;

    // scene.AddSolid(mirr);
    scene.LoadCornell(2);
    // scene.AddSolid(std::make_shared<Sphere>(1, Vec(-1, 0, 0), Vec(), Vec(1,1,1), Surface::REFR));
    scene.LoadOBJModel("dodecahedron.txt");
    scene.TakePicture(0);

    return 0;
}