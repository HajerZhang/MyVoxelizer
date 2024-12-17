#ifndef __KERNEL_CU__
#define __KERNEL_CU__

#include <Voxelizer.h>
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
        Vector3d minTri = triangle.min();
        Vector3d maxTri = triangle.max();
        Vector3d minGrid = Vector3d(
            std::max(0, (int)std::floor((minTri.x - d_minGridX) / d_voxelSizeX)),
            std::max(0, (int)std::floor((minTri.y - d_minGridY) / d_voxelSizeY)),
            std::max(0, (int)std::floor((minTri.z - d_minGridZ) / d_voxelSizeZ))
        );
        Vector3d maxGrid = Vector3d(
            std::min(d_numX - 1, (int)std::ceil((maxTri.x - d_minGridX) / d_voxelSizeX)),
            std::min(d_numY - 1, (int)std::ceil((maxTri.y - d_minGridY) / d_voxelSizeY)),
            std::min(d_numZ - 1, (int)std::ceil((maxTri.z - d_minGridZ) / d_voxelSizeZ))
        );

        for(int z = minGrid.z; z <= maxGrid.z; z++){
            for(int y = minGrid.y; y <= maxGrid.y; y++){
                for(int x = minGrid.x; x <= maxGrid.x; x++){
                    Vector3d minVoxel = Vector3d(
                        d_minGridX + x * d_voxelSizeX,
                        d_minGridY + y * d_voxelSizeY,
                        d_minGridZ + z * d_voxelSizeZ
                    );
                    Vector3d maxVoxel = Vector3d(
                        d_minGridX + (x + 1) * d_voxelSizeX,
                        d_minGridY + (y + 1) * d_voxelSizeY,
                        d_minGridZ + (z + 1) * d_voxelSizeZ
                    );

                    

                }
            }
        }
    }  
}

#endif