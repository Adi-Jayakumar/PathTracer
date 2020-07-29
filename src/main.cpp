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
    Camera CamUp = Camera(2, PTUtility::W, PTUtility::H, Vec(0, 0, -1.99), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    scene.AddCamera(CamUp);
    
    scene.LoadCornell(2);
    scene.LoadOBJModel("dodecahedron.txt");
    scene.TakePicture(0);

    return 0;
}