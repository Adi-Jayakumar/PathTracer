#include "scene.h"

void Scene::AddSolid(Solid s)
{
    objects.push_back(s);
}

void Scene::AddCamera(Camera c)
{
    cameras.push_back(c);
}

HitRecord Scene::ClosestIntersection(Ray r)
{
    // Assume no hit initally
    double recordT = std::numeric_limits<double>::max();
    int idMin = -1;
    double t;

    // iterate through and check for a hit
    for (long unsigned i = 0; i < objects.size(); i++)
    {
        t = std::numeric_limits<double>::max();

        // if hit then set the "record" values accordingly
        if (objects[i].shape->Intersect(r, t) && t < recordT)
        {
            recordT = t;
            idMin = i;
        }
    }

    // return a hitrecord with the details
    return HitRecord(recordT, idMin);
}

Vec Scene::RayColour(Ray r, int depth)
{
    // if it has bounced enough times return black
    if (depth > PTUtility::MaxDepth)
        return Vec();

    HitRecord rec = ClosestIntersection(r);

    // if no sphere was hit return black
    if (rec.id == -1)
        return Vec();

    // the object that was hit
    Solid hit = objects[rec.id];
    // the hit-point
    Vec hitPt = r.o + r.d * rec.t;
    // the surface normal
    Vec normal = hit.shape->Normal(hitPt);
    // the normal that faces toward the ray
    Vec rayNormal = Vec::Dot(normal, r.d) < 0 ? normal : normal * -1;

    Vec albedo = hit.c;

    // probability of terminating early
    double finProb = PTUtility::Luma(albedo);
    if (++depth > 5)
    {
        if (PTUtility::Random() < finProb)
            albedo = albedo / finProb;
        else
            return hit.e;
    }

    if (hit.s == Surface::DIFF)
    {
        double phi = 2 * PTUtility::PI * PTUtility::Random();
        double r2 = PTUtility::Random();
        double sinTheta = sqrt(r2);
        double cosTheta = sqrt(1 - r2);

        // building basis for R3 of perpendicular vectors around the normal facing outwards
        Vec w = rayNormal.Norm();
        // Vec u = (Vec::Cross(fabs(w.x) > .1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w)).Norm();
        Vec u = Vec(rayNormal.z, rayNormal.z, -rayNormal.x - 2 * rayNormal.y).Norm();
        Vec v = Vec::Cross(u, w).Norm();

        Vec newDir = (u * cos(phi) * sinTheta + v * sin(phi) * sinTheta + w * cosTheta).Norm();
        return hit.e + albedo * RayColour(Ray(hitPt, newDir), depth);
    }
    else if (hit.s == Surface::SPEC)
    {
        Vec reflectedDir = r.d - normal * 2 * Vec::Dot(normal, r.d);
        return hit.e + albedo * RayColour(Ray(hitPt, reflectedDir), depth);
    }
    else if (hit.s == Surface::SPECGLOSS)
    {
        // choose a random ray direction sampling in a cone around the true reflected direction
        Vec specReflDir = r.d - normal * 2 * Vec::Dot(normal, r.d);
        Vec w = specReflDir.Norm();
        Vec u = Vec(w.z, w.z, -w.x - 2 * w.y).Norm();
        Vec v = Vec::Cross(u, w).Norm();

        double rad = 0.125 * sqrt(PTUtility::Random());
        double theta = PTUtility::Random() * 2 * PTUtility::PI;

        double xPrime = rad * cos(theta);
        double yPrime = rad * sin(theta);

        Vec newDir = (u * xPrime + v * yPrime + w).Norm();
        return hit.e + albedo * RayColour(Ray(hitPt, newDir), depth);
    }
    else if (hit.s == Surface::REFRGLOSS)
    {
        // perfect reflected direction whcih we now need to jitter
        Vec specReflDir = r.d - normal * 2 * Vec::Dot(normal, r.d);
        Vec w = specReflDir.Norm();
        Vec u = Vec(w.z, w.z, -w.x - 2 * w.y).Norm();
        Vec v = Vec::Cross(u, w).Norm();

        double rad = 0.125 * sqrt(PTUtility::Random());
        double theta = PTUtility::Random() * 2 * PTUtility::PI;

        double xPrime = rad * cos(theta);
        double yPrime = rad * sin(theta);

        // jittered reflected direction
        Vec reflDir = (u * xPrime + v * yPrime + w).Norm();
        // jittered refrlected ray
        Ray reflRay = Ray(hitPt, reflDir);

        // is the ray going into the object or out?
        bool isInto = Vec::Dot(normal, rayNormal) > 0;

        // refractive indices
        double n1 = 1;
        double n2 = 1.5;
        double netN = isInto ? n1 / n2 : n2 / n1;

        // cosines of the angles
        double cosTheta = Vec::Dot(r.d, rayNormal);
        double cosTheta2Sqr = 1 - netN * netN * (1 - cosTheta * cosTheta);

        // total internal reflection
        if (cosTheta2Sqr < 0)
            return hit.e + albedo * RayColour(reflRay, depth);

        // perfect refracted direction which we now need to jitter
        Vec specRefrDir = (r.d * netN - normal * ((isInto ? 1 : -1) * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();
        w = specRefrDir.Norm();
        u = Vec(w.z, w.z, -w.x - 2 * w.y).Norm();
        v = Vec::Cross(u, w).Norm();

        rad = 0.03125 * sqrt(PTUtility::Random());
        theta = PTUtility::Random() * 2 * PTUtility::PI;

        xPrime = rad * cos(theta);
        yPrime = rad * sin(theta);

        // jittered refracted direction
        Vec refrDir = (u * xPrime + v * yPrime + w).Norm();
        // jittered refracted ray
        Ray refrRay = Ray(hitPt, refrDir);

        double a = n2 - n1;
        double b = n1 + n2;
        double R0 = (a * a) / (b * b);

        double cosTheta2 = Vec::Dot(refrDir, normal);
        double c = 1 - (isInto ? -cosTheta : cosTheta2);
        double refl = R0 + (1 - R0) * c * c * c * c;
        double refr = 1 - refl;

        double P = 0.25 + 0.5 * refl;

        double reflectionWeight = refl / P;
        double refractionWeight = refr / (1 - P);

        if (depth < 3)
            return hit.e + albedo * (RayColour(reflRay, depth) * refl + RayColour(refrRay, depth) * refr);
        else
        {
            if (PTUtility::Random() < P)
                return hit.e + albedo * RayColour(reflRay, depth) * reflectionWeight;
            else
                return hit.e + albedo * RayColour(refrRay, depth) * refractionWeight;
        }
    }
    else if (hit.s == Surface::REFR)
    {
        Ray reflectedRay = Ray(hitPt, r.d - normal * 2 * Vec::Dot(normal, r.d));

        // is the ray going into the object or out?
        bool isInto = Vec::Dot(normal, rayNormal) > 0;

        // refractive indices
        double n1 = 1;
        double n2 = 1.5;
        double netN = isInto ? n1 / n2 : n2 / n1;

        // cosines of the angles
        double cosTheta = Vec::Dot(r.d, rayNormal);
        double cosTheta2Sqr = 1 - netN * netN * (1 - cosTheta * cosTheta);

        // total internal reflection
        if (cosTheta2Sqr < 0)
            return hit.e + albedo * RayColour(reflectedRay, depth);

        Vec refractedDir = (r.d * netN - normal * ((isInto ? 1 : -1) * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();
        Ray refractedRay = Ray(hitPt, refractedDir);
        // Vec refractedDir = (r.d * netN - rayNormal * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();

        // approximating reflection and refraction weights
        double a = n2 - n1;
        double b = n1 + n2;
        double R0 = (a * a) / (b * b);

        double cosTheta2 = Vec::Dot(refractedDir, normal);
        double c = 1 - (isInto ? -cosTheta : cosTheta2);
        double refl = R0 + (1 - R0) * c * c * c * c;
        double refr = 1 - refl;

        double P = 0.25 + 0.5 * refl;

        double reflectionWeight = refl / P;
        double refractionWeight = refr / (1 - P);

        if (depth < 3)
            return hit.e + albedo * (RayColour(reflectedRay, depth) * refl + RayColour(refractedRay, depth) * refr);
        else
        {
            if (PTUtility::Random() < P)
                return hit.e + albedo * RayColour(reflectedRay, depth) * reflectionWeight;
            else
                return hit.e + albedo * RayColour(refractedRay, depth) * refractionWeight;
        }
    }
    return Vec();
}

void Scene::TakePicture(int index)
{
    Vec *image = new Vec[PTUtility::W * PTUtility::H];
    Image im = Image(PTUtility::W, PTUtility::H, index);
    Ray r;
    int rowCount = 0;

    int numRays = PTUtility::W * PTUtility::H * PTUtility::NumSamps * PTUtility::SubPixSize * PTUtility::SubPixSize;

    std::cout << std::endl
              << std::endl
              << "Started casting rays" << std::endl
              << "Number of rays cast: " << numRays << " (~10^" << floor(log10(numRays)) << ")" << std::endl
              << std::endl;
#pragma omp parallel for schedule(dynamic, 1) private(r)
    // image rows
    for (int i = 0; i < PTUtility::H; i++)
    {
        // image cols
        for (int j = 0; j < PTUtility::W; j++)
        {
            Vec c;
            // sub pixel rows
            for (int sy = 0; sy < PTUtility::SubPixSize; sy++)
            {
                // sub pixel cols
                for (int sx = 0; sx < PTUtility::SubPixSize; sx++)
                {
                    // sampling the pixel
                    for (int s = 0; s < PTUtility::NumSamps; s++)
                    {
                        r = cameras[index].GenerateRay(i, j, sx, sy);
                        c = c + RayColour(r, 0);
                    }
                }
            }
            image[i * PTUtility::W + j] = c / ((double)PTUtility::NumSamps * PTUtility::SubPixSize * PTUtility::SubPixSize);
        }
        rowCount++;
        if (rowCount % (PTUtility::H / 10) == 0)
            std::cout << "Progress: " << (static_cast<double>(rowCount) / PTUtility::H) * 100 << "%" << std::endl;
    }
    im.Set(image);
    delete image;
}

void Scene::LoadCornell(double boxSize)
{
    Camera cam = Camera(boxSize * 2, PTUtility::W, PTUtility::H, Vec(0, 0, -boxSize + 0.001), Vec(0, 0, 1), Vec(0, 1, 0), PTUtility::PI / 3.0);
    cameras.emplace_back(cam);
    std::shared_ptr<Plane> front = std::make_shared<Plane>(Vec(0, 0, -1), Vec(0, 0, boxSize));
    std::shared_ptr<Plane> right = std::make_shared<Plane>(Vec(-1, 0, 0), Vec(boxSize, 0, 0));
    std::shared_ptr<Plane> back = std::make_shared<Plane>(Vec(0, 0, 1), Vec(0, 0, -boxSize));
    std::shared_ptr<Plane> left = std::make_shared<Plane>(Vec(1, 0, 0), Vec(-boxSize, 0, 0));
    std::shared_ptr<Plane> top = std::make_shared<Plane>(Vec(0, -1, 0), Vec(0, boxSize, 0));
    std::shared_ptr<Plane> bottom = std::make_shared<Plane>(Vec(0, 1, 0), Vec(0., -boxSize, 0));
    objects.emplace_back(Solid(front, Vec(), Vec(1, 1, 1), Surface::DIFF));
    objects.emplace_back(Solid(right, Vec(), Vec(0, 1, 0), Surface::DIFF));
    objects.emplace_back(Solid(back, Vec(), Vec(), Surface::DIFF));
    objects.emplace_back(Solid(left, Vec(), Vec(1, 0, 0), Surface::DIFF));
    objects.emplace_back(Solid(top, Vec(1, 1, 1), Vec(), Surface::DIFF));
    objects.emplace_back(Solid(bottom, Vec(), Vec(1, 1, 1), Surface::DIFF));
}

void Scene::LoadOBJModel(std::string fPath)
{
    std::ifstream file(fPath);
    std::vector<Vec> vertices;

    while (!file.eof())
    {
        char line[1024];
        file.getline(line, 1024);
        std::stringstream s;
        s << line;
        char junk;
        if (line[0] == 'v')
        {
            double x, y, z;
            s >> junk >> x >> y >> z;
            Vec v = Vec(x, y, z);
            vertices.push_back(v);
        }
        else if (line[0] == 'f')
        {
            int v1, v2, v3;
            s >> junk >> v1 >> v2 >> v3;
            std::shared_ptr<Shape> temp = std::make_shared<Triangle>(vertices[v1 - 1] - Vec(1.25, 1.25, 1.25), vertices[v2 - 1] - Vec(1.25, 1.25, 1.25), vertices[v3 - 1] - Vec(1.25, 1.25, 1.25));
            AddSolid(Solid(temp, Vec(), Vec(1, 1, 1), Surface::REFR));
        }
    }
    file.close();
}