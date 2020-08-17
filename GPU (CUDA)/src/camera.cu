#include "camera.cuh"

__device__ Camera::Camera()
{
    loc = Vec();
    forward = Vec();
    up = Vec();
    right = Vec();
    focus = Vec();
    worldH = 0;
    worldW = 0;
    hFOV = 0;
    nPixelsX = 0;
    nPixelsY = 0;
}

__device__ Camera::Camera(double _worldW, int _nPixelsX, int _nPixelsY, Vec _loc, Vec _forward, Vec _up, double _hFOV)
{
    worldW = _worldW;
    nPixelsX = _nPixelsX;
    nPixelsY = _nPixelsY;
    worldH = worldW * (double)nPixelsY / nPixelsX;
    loc = _loc;
    forward = _forward.Norm();
    up = _up.Norm();
    hFOV = _hFOV;
    right = Vec::Cross(up, forward).Norm();
    focus = _loc - forward * (worldW / (2 * tan(hFOV / 2)));

    // std::cout << "worldW " << worldW << std::endl;
    // std::cout << "worldH " << worldH << std::endl;
    // std::cout << "nPixelsX " << nPixelsX << std::endl;
    // std::cout << "nPixelsY " << nPixelsY << std::endl;
    // std::cout << "forward " << forward << std::endl;
    // std::cout << "up " << up << std::endl;
    // std::cout << "hFOV " << hFOV << std::endl;
    // std::cout << "right " << right << std::endl;
    // std::cout << "d " << worldW / (2 * tan(hFOV / 2)) << std::endl;
    // std::cout << "focus" << focus << std::endl;
};

__device__ Ray Camera::GenerateAARay(int i, int j, int sx, int sy, curandState &state)
{
    /*
    0 <= j < nPixelsX, 0 <= i < nPixelsY
    o----------------> j increasing
    |   o--- 
    |    \ |
    |     \|
    |      c
    |
    |
    |
    v
    i increasing

    c is the centre of the screen, o is some point on it

    The point o is at (j, i) in the image and once i and j are normalised to be between 0 and 1,
    we can see that with respect to c, o becomes (j - 0.5, 0.5 - i) in a world where the top of 
    the screen is up and the right is right
    */
    double r1 = 2 * PTUtility::Random(state);
    double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
    double r2 = 2 * PTUtility::Random(state);
    double dy = r1 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
    Vec deltaRight = right * worldW * (((double)j + (sx + 0.5 + dx) / 2) / nPixelsX - 0.5);
    Vec deltaUp = up * worldH * (0.5 - ((double)i + (sy + 0.5 + dy) / 2) / nPixelsY);
    Vec rayOr = deltaUp + deltaRight + loc;
    return Ray(rayOr, (rayOr - focus).Norm());
}

__device__ Ray Camera::GenerateRay(int i, int j)
{
    Vec deltaRight = right * worldW * ((double)j / nPixelsX - 0.5);
    Vec deltaUp = up * worldH * (0.5 - (double)i / nPixelsY);
    Vec rayOr = deltaUp + deltaRight + loc;
    return Ray(rayOr, (rayOr - focus).Norm());
}