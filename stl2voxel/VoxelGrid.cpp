////////////////////////////////////////////////////////////////////////////////////////    
// Copyright (c) 2024 Hajer Zhang, IDEAS, DLUT.
//  
// Permission is hereby granted, free of charge, to any person obtaining a copy  of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions:  
//  
// The above copyright notice and this permission notice shall be included in all  
// copies or substantial portions of the Software.  
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A  
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF  
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE  
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  
//  
// Author: Hajer Zhang 
// Date: 2024-12-17 
// Description: A voxelization library to process STL files into voxel grids, supporting 
//              academic, commercial, and various other purposes. Contributions and 
//              citations are welcome.  
//////////////////////////////////////////////////////////////////////////////////////

#include <Voxelizer.h>
#include <cmath>
using namespace voxel;

VoxelGrid::VoxelGrid()
{   
    m_numVoxels = 0;
    m_numX = 0;
    m_numY = 0;
    m_numZ = 0;
    m_voxelSize = Vector3d(0, 0, 0);
    m_minGrid = Vector3d(0, 0, 0);
    m_maxGrid = Vector3d(0, 0, 0);
    m_grid.clear();
}

VoxelGrid::~VoxelGrid()
{
    m_grid.clear();
    m_grid.shrink_to_fit();
}

VoxelGrid::VoxelGrid(
    const STLMesh *stlmesh,
    const int numX, const int numY, const int numZ
)
{
    m_numVoxels = numX * numY * numZ;
    m_numX = numX;
    m_numY = numY;
    m_numZ = numZ;

    m_minGrid = Vector3d(stlmesh->triangleList[0].v0.x, stlmesh->triangleList[0].v0.y, stlmesh->triangleList[0].v0.z);
    m_maxGrid = Vector3d(stlmesh->triangleList[0].v0.x, stlmesh->triangleList[0].v0.y, stlmesh->triangleList[0].v0.z);

    for(auto triangle : stlmesh->triangleList)
    {
        if(triangle.v0.x < m_minGrid.x) m_minGrid.x = triangle.v0.x;
        if(triangle.v0.y < m_minGrid.y) m_minGrid.y = triangle.v0.y;
        if(triangle.v0.z < m_minGrid.z) m_minGrid.z = triangle.v0.z;

        if(triangle.v0.x > m_maxGrid.x) m_maxGrid.x = triangle.v0.x;
        if(triangle.v0.y > m_maxGrid.y) m_maxGrid.y = triangle.v0.y;
        if(triangle.v0.z > m_maxGrid.z) m_maxGrid.z = triangle.v0.z;
    }

    m_voxelSize = Vector3d(
        (m_maxGrid.x - m_minGrid.x) / numX,
        (m_maxGrid.y - m_minGrid.y) / numY,
        (m_maxGrid.z - m_minGrid.z) / numZ
    );

    // Expand the grid to avoid the boundary problem
    m_minGrid.x -= 3*m_voxelSize.x;
    m_minGrid.y -= 3*m_voxelSize.y;
    m_minGrid.z -= 3*m_voxelSize.z;
    m_maxGrid.x += 3*m_voxelSize.x;
    m_maxGrid.y += 3*m_voxelSize.y;
    m_maxGrid.z += 3*m_voxelSize.z;
    m_voxelSize = Vector3d(
        (m_maxGrid.x - m_minGrid.x) / numX,
        (m_maxGrid.y - m_minGrid.y) / numY,
        (m_maxGrid.z - m_minGrid.z) / numZ
    );

    m_grid.clear();
    m_grid.resize(m_numVoxels, -1);
}

void VoxelGrid::MarkSurfaceVoxels(const Vector3d &voxelCoord){
    m_grid[voxelCoord.z * m_numX * m_numY + voxelCoord.y * m_numX + voxelCoord.x] = 0;
}

void VoxelGrid::MarkInsideVoxels(const Vector3d &voxelCoord){
    m_grid[voxelCoord.z * m_numX * m_numY + voxelCoord.y * m_numX + voxelCoord.x] = 1;
}

bool aabbCheck(const Vector3d &minTri, const Vector3d &maxTri, const Vector3d &minVoxel, const Vector3d &maxVoxel)
{
    if(minTri.x > maxVoxel.x || maxTri.x < minVoxel.x) return false;
    if(minTri.y > maxVoxel.y || maxTri.y < minVoxel.y) return false;
    if(minTri.z > maxVoxel.z || maxTri.z < minVoxel.z) return false;
    return true;
}

void VoxelGrid::ComfirmSurfaceVoxels(const STLMesh *stlmesh)
{   
    # pragma omp parallel for
    for(const auto& triangle : stlmesh->triangleList)
    {   
        Vector3d minTri = triangle.min();
        Vector3d maxTri = triangle.max();
        Vector3d minGrid = Vector3d(
            std::max(0, (int)std::floor((minTri.x - m_minGrid.x) / m_voxelSize.x)),
            std::max(0, (int)std::floor((minTri.y - m_minGrid.y) / m_voxelSize.y)),
            std::max(0, (int)std::floor((minTri.z - m_minGrid.z) / m_voxelSize.z))
        );
        Vector3d maxGrid = Vector3d(
            std::min(m_numX - 1, (int)std::ceil((maxTri.x - m_minGrid.x) / m_voxelSize.x)),
            std::min(m_numY - 1, (int)std::ceil((maxTri.y - m_minGrid.y) / m_voxelSize.y)),
            std::min(m_numZ - 1, (int)std::ceil((maxTri.z - m_minGrid.z) / m_voxelSize.z))
        );

        for(int z = minGrid.z; z <= maxGrid.z; z++){
            for(int y = minGrid.y; y <= maxGrid.y; y++){
                for(int x = minGrid.x; x <= maxGrid.x; x++){
                    Vector3d minVoxel = Vector3d(
                        m_minGrid.x + x * m_voxelSize.x,
                        m_minGrid.y + y * m_voxelSize.y,
                        m_minGrid.z + z * m_voxelSize.z
                    );
                    Vector3d maxVoxel = Vector3d(
                        m_minGrid.x + (x + 1) * m_voxelSize.x,
                        m_minGrid.y + (y + 1) * m_voxelSize.y,
                        m_minGrid.z + (z + 1) * m_voxelSize.z
                    );

                    if(aabbCheck(minTri, maxTri, minVoxel, maxVoxel))
                    {
                        MarkSurfaceVoxels(Vector3d(x, y, z));
                    }

                }
            }
        }
    }
}

void VoxelGrid::ComfirmInsideVoxels(const STLMesh *stlmesh)
{   
    # pragma omp parallel for
    for(int z = 0; z < m_numZ; z++){
        for(int y = 0; y < m_numY; y++){
            for(int x = 0; x < m_numX; x++){
                if(m_grid[z * m_numX * m_numY + y * m_numX + x] == 0) continue;
                Vector3d voxelCoord = Vector3d(x, y, z);
                int surfaceCount = 0;
                
                // Check the 6 directions if there is a surface voxel
                // x positive direction
                for(int i = x + 1; i < m_numX; i++){
                    if(m_grid[z * m_numX * m_numY + y * m_numX + i] == 0){
                        surfaceCount++;
                        break;
                    }
                }
                // x negative direction
                for(int i = x - 1; i >= 0; i--){
                    if(m_grid[z * m_numX * m_numY + y * m_numX + i] == 0){
                        surfaceCount++;
                        break;
                    }
                }
                // y positive direction
                for(int i = y + 1; i < m_numY; i++){
                    if(m_grid[z * m_numX * m_numY + i * m_numX + x] == 0){
                        surfaceCount++;
                        break;
                    }
                }
                // y negative direction
                for(int i = y - 1; i >= 0; i--){
                    if(m_grid[z * m_numX * m_numY + i * m_numX + x] == 0){
                        surfaceCount++;
                        break;
                    }
                }
                // z positive direction
                for(int i = z + 1; i < m_numZ; i++){
                    if(m_grid[i * m_numX * m_numY + y * m_numX + x] == 0){
                        surfaceCount++;
                        break;
                    }
                }
                // z negative direction
                for(int i = z - 1; i >= 0; i--){
                    if(m_grid[i * m_numX * m_numY + y * m_numX + x] == 0){
                        surfaceCount++;
                        break;
                    }
                }

                bool inside = surfaceCount == 6 ? true : false;

                if(inside) MarkInsideVoxels(voxelCoord);
            }
        }
    }
}

void VoxelGrid::Update(const STLMesh *stlmesh)
{
    ComfirmSurfaceVoxels(stlmesh);
    ComfirmInsideVoxels(stlmesh);
}

void VoxelGrid::OutputVTKFile(const std::string &outputfile){
    std::ofstream file(outputfile);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    file << "# vtk DataFile Version 3.0" << std::endl;
    file << "Voxel Grid" << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET STRUCTURED_POINTS" << std::endl;
    file << "DIMENSIONS " << m_numX + 1 << " " << m_numY + 1 << " " << m_numZ + 1 << std::endl;
    file << "SPACING " << m_voxelSize.x << " " << m_voxelSize.y << " " << m_voxelSize.z << std::endl;
    file << "ORIGIN " << m_minGrid.x << " " << m_minGrid.y << " " << m_minGrid.z << std::endl;
    file << "CELL_DATA " << m_numVoxels << std::endl;
    file << "SCALARS cell_type double" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < m_numVoxels; i++)
    {
        file << double(m_grid[i]) << std::endl;
    }

    file.close();
}