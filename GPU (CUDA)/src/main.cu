#include "camera.cuh"
#include "cuda_device_runtime_api.h"
#include "hitrecord.cuh"
#include "image.cuh"
#include "ptutility.cuh"
#include "scene.cuh"
#include "solid.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "vector.cuh"
#include <curand_kernel.h>
#include <iostream>
#include <time.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void CreateWorld(Solid **objects)
{
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //     *objects = new Solid(new Sphere(2, Vec(-2, 0, 3)), Vec(), Vec(1, 1, 1), Surface::SPEC);
    //     *(objects + 1) = new Solid(new Sphere(2, Vec(2, 0, 3)), Vec(1, 1, 1), Vec(), Surface::DIFF);
    //     *(objects + 2) = new Solid(new Sphere(20, Vec()), Vec(), Vec(1, 1, 1), Surface::DIFF);
    //     // *(objects + 1) = new Solid(new Sphere(2, Vec(2, 0, 3)), Vec(), Vec(1, 1, 1), Surface::DIFF);
    // }
    
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        Plane* front = new Plane(Vec(0, 0, -1), Vec(0, 0, 10));
        Plane* right = new Plane(Vec(-1, 0, 0), Vec(10, 0, 0));
        Plane* back = new Plane(Vec(0, 0, 1), Vec(0, 0, -10));
        Plane* left = new Plane(Vec(1, 0, 0), Vec(-10, 0, 0));
        Plane* top = new Plane(Vec(0, -1, 0), Vec(0, 10, 0));
        Plane* bottom = new Plane(Vec(0, 1, 0), Vec(0., -10, 0));
        
        *(objects) = new Solid(front, Vec(), Vec(1, 1, 1), Surface::DIFF);
        *(objects + 1) = new Solid(right, Vec(), Vec(0, 1, 0), Surface::DIFF);
        *(objects + 2) = new Solid(back, Vec(), Vec(), Surface::DIFF);
        *(objects + 3) = new Solid(left, Vec(), Vec(1, 0, 0), Surface::DIFF);
        *(objects + 4) = new Solid(top, Vec(1, 1, 1), Vec(), Surface::DIFF);
        *(objects + 5) = new Solid(bottom, Vec(), Vec(1, 1, 1), Surface::DIFF);
        
        *(objects + 6) = new Solid(new Sphere(5, Vec(-5, -5, 5)), Vec(), Vec(1,1,1), Surface::SPECGLOSS);
        // *(objects + 6) = new Solid(new Sphere(5, Vec()), Vec(), Vec(1,1,1), Surface::REFR);
    }
    
}

__device__ Vec Colour(Ray r, Solid ** objects, int nObj)
{
    
    HitRecord rec = Scene::ClosestIntersection(r, objects, nObj);
    if(rec.id == -1)
        return Vec();
    else
        return objects[rec.id]->c;
    
}

__global__ void FreeWorld(Solid **objects, int nObj)
{
    for (int i = 0; i < nObj; i++)
    {
        delete *(objects + i);
    }
}

__global__ void Render(Vec *fb, Solid **objects, int nObj)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= PTUtility::W) || (j >= PTUtility::H))
        return;
    else
    {
        int index = j * PTUtility::W + i;
        curandState state;
        curand_init(1234, index, 0, &state);
        Camera cam = Camera(20, PTUtility::W, PTUtility::H, Vec(0,0,-10), Vec(0, 0, 1), Vec(0, 1, 0), 3.1415926535 / 3);
        Ray r = Ray();
        Vec c = Vec();
        for (int sx = 0; sx < PTUtility::SubPixSize; sx++)
        {
            for (int sy = 0; sy < PTUtility::SubPixSize; sy++)
            {
                for (int s = 0; s < PTUtility::NumSamps; s++)
                {
                    r = cam.GenerateAARay(j, i, sx, sy, state);
                    // c = c + Scene::RayColour(r, 0, objects, state, nObj);
                    c = c + Scene::RayColour(r, objects, state, nObj);
                    // c = c + Colour(r, objects, nObj);
                }
            }
        }
        fb[index] = c / ((double)PTUtility::NumSamps * PTUtility::SubPixSize * PTUtility::SubPixSize);
    }
}

int main()
{
    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << PTUtility::W << "x" << PTUtility::H << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = PTUtility::W * PTUtility::H;
    size_t fb_size = num_pixels * sizeof(Vec);
    int nObj = 7;

    Solid **objects;
    checkCudaErrors(cudaMalloc((void **)&objects, sizeof(Solid *)));
    CreateWorld<<<1, 1>>>(objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate FB
    Vec *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(PTUtility::W / tx + 1, PTUtility::H / ty + 1);
    dim3 threads(tx, ty);
    Render<<<blocks, threads>>>(fb, objects, nObj);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    Image image = Image(PTUtility::W, PTUtility::H, 0);
    image.Set(fb);
    checkCudaErrors(cudaFree(fb));
    FreeWorld<<<1,1>>>(objects, nObj);
    return 0;
}