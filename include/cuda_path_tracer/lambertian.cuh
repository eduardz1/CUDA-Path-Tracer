#pragma once

class Lambertian{

    public:
    __host__ __device__ Lambertian(const Vec3 albedo) : albedo(albedo) {}

    __device__ bool scatter(const Ray &ray, Vec3 &normal, Vec3 &point, Vec3 &attenuation, Ray &scattered, curandState &state){
        auto const scatter_direction = normal + vectorOnHemisphere(normal, state);
        scattered = Ray(point, scatter_direction);
        attenuation = albedo;
        return true;
    }

    private:
        Vec3 albedo;

};

