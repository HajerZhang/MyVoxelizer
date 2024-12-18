#ifndef __CUMATH_CU__
#define __CUMATH_CU__

__forceinline__ __device__ __host__ int cumax(int a, int b) {
    return (a > b) ? a : b;
}

__forceinline__ __device__ __host__ int cumin(int a, int b) {
    return (a < b) ? a : b;
}

template <typename T>
__forceinline__ __device__ __host__ T cufloor(T x) {
    return (x >= 0.0f) ? (int)x : (int)x - 1;
}

template <typename T>
__forceinline__ __device__ __host__ T cuceil(T x) {
    return (x == (int)x) ? x : ((x > 0.0f) ? (int)x + 1 : (int)x);
}

#endif 