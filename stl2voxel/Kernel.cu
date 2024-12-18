#ifndef __KERNEL_CU__
#define __KERNEL_CU__

#include <Voxelizer.cuh>
#include <cumath.cuh>
using namespace voxel;

#define BLOCKX 8
#define BLOCKY 8
#define BLOCKZ 8

__constant__ int d_numX;
__constant__ int d_numY;
__constant__ int d_numZ;
__constant__ double d_voxelSizeX;
__constant__ double d_voxelSizeY;
__constant__ double d_voxelSizeZ;
__constant__ double d_minGridX;
__constant__ double d_minGridY;
__constant__ double d_minGridZ;

__global__ void ComfirmSurfaceVoxelsKernel(
    const Triangle *d_triangleList, const int numTri, int *d_grid
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int active = idx < numTri;
    
    if(active){
        Triangle triangle = d_triangleList[idx];
        dim3 minTri = dim3(
            cufloor(cumin(triangle.v0.x, cumin(triangle.v1.x, triangle.v2.x))),
            cufloor(cumin(triangle.v0.y, cumin(triangle.v1.y, triangle.v2.y))),
            cufloor(cumin(triangle.v0.z, cumin(triangle.v1.z, triangle.v2.z)))
        );
        dim3 maxTri = dim3(
            cuceil(cumax(triangle.v0.x, cumax(triangle.v1.x, triangle.v2.x))),
            cuceil(cumax(triangle.v0.y, cumax(triangle.v1.y, triangle.v2.y))),
            cuceil(cumax(triangle.v0.z, cumax(triangle.v1.z, triangle.v2.z)))
        );
        dim3 minGrid = dim3(
            cumin(0, (int)cufloor((minTri.x - d_minGridX) / d_voxelSizeX)),
            cumin(0, (int)cufloor((minTri.y - d_minGridY) / d_voxelSizeY)),
            cumin(0, (int)cufloor((minTri.z - d_minGridZ) / d_voxelSizeZ))
        );
        dim3 maxGrid = dim3(
            cumin(d_numX - 1, (int)cuceil((maxTri.x - d_minGridX) / d_voxelSizeX)),
            cumin(d_numY - 1, (int)cuceil((maxTri.y - d_minGridY) / d_voxelSizeY)),
            cumin(d_numZ - 1, (int)cuceil((maxTri.z - d_minGridZ) / d_voxelSizeZ))
        );

        for(int z = minGrid.z; z <= maxGrid.z; z++){
            for(int y = minGrid.y; y <= maxGrid.y; y++){
                for(int x = minGrid.x; x <= maxGrid.x; x++){
                    dim3 minVoxel = dim3(
                        d_minGridX + x * d_voxelSizeX,
                        d_minGridY + y * d_voxelSizeY,
                        d_minGridZ + z * d_voxelSizeZ
                    );
                    dim3 maxVoxel = dim3(
                        d_minGridX + (x + 1) * d_voxelSizeX,
                        d_minGridY + (y + 1) * d_voxelSizeY,
                        d_minGridZ + (z + 1) * d_voxelSizeZ
                    );

                    if(minTri.x > maxVoxel.x || maxTri.x < minVoxel.x) continue;
                    if(minTri.y > maxVoxel.y || maxTri.y < minVoxel.y) continue;
                    if(minTri.z > maxVoxel.z || maxTri.z < minVoxel.z) continue;

                    d_grid[z * d_numX * d_numY + y * d_numX + x] = 0;
                }
            }
        }
    }  
    __syncthreads();
}

__global__ void ComfirmInsideVoxelsKernel(int *d_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int active = idx < d_numX && idy < d_numY && idz < d_numZ;

    if(active && d_grid[idz * d_numX * d_numY + idy * d_numX + idx] == -1){
        int idxGrid = idz * d_numX * d_numY + idy * d_numX + idx;
        int surfaceCount = 0;

        // Check 6 directions if there is a surface voxel
        // x direction
        for(int i = idx + 1; i < d_numX; i++){
            if(d_grid[idz * d_numX * d_numY + idy * d_numX + i] == 0){
                surfaceCount++;
                break;
            }
        }
        // -x direction
        for(int i = idx - 1; i >= 0; i--){
            if(d_grid[idz * d_numX * d_numY + idy * d_numX + i] == 0){
                surfaceCount++;
                break;
            }
        }
        // y direction
        for(int i = idy + 1; i < d_numY; i++){
            if(d_grid[idz * d_numX * d_numY + i * d_numX + idx] == 0){
                surfaceCount++;
                break;
            }
        }
        // -y direction
        for(int i = idy - 1; i >= 0; i--){
            if(d_grid[idz * d_numX * d_numY + i * d_numX + idx] == 0){
                surfaceCount++;
                break;
            }
        }
        // z direction
        for(int i = idz + 1; i < d_numZ; i++){
            if(d_grid[i * d_numX * d_numY + idy * d_numX + idx] == 0){
                surfaceCount++;
                break;
            }
        }
        // -z direction
        for(int i = idz - 1; i >= 0; i--){
            if(d_grid[i * d_numX * d_numY + idy * d_numX + idx] == 0){
                surfaceCount++;
                break;
            }
        }

        if(surfaceCount == 6){
            d_grid[idxGrid] = 1;
        }

    }
    __syncthreads();
}

#endif