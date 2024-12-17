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
////////////////////////////////////////////////////////////////////////////////////////  

#ifndef __VOXELIZER_H__
#define __VOXELIZER_H__

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace voxel{
    struct Vector3d
    {
        double x, y, z;
        Vector3d(double x, double y, double z) : x(x), y(y), z(z) {}
        Vector3d() : x(0), y(0), z(0) {}
        ~Vector3d() {}
        int operator[](int i) const {
            if(i == 0) return x;
            if(i == 1) return y;
            if(i == 2) return z;
            throw std::runtime_error("Index out of range");
            exit(1);
        }
    };

    struct Triangle
    {
        Vector3d normal;
        Vector3d v0;
        Vector3d v1;
        Vector3d v2;
        ~Triangle() {}
        Vector3d min() const {
            Vector3d min = v0;
            min.x = std::min(v0.x, std::min(v1.x, v2.x));
            min.y = std::min(v0.y, std::min(v1.y, v2.y));
            min.z = std::min(v0.z, std::min(v1.z, v2.z));
            return min;
        }
        Vector3d max() const {
            Vector3d max = v0;
            max.x = std::max(v0.x, std::max(v1.x, v2.x));
            max.y = std::max(v0.y, std::max(v1.y, v2.y));
            max.z = std::max(v0.z, std::max(v1.z, v2.z));
            return max;
        }
        int mainDirect() const {
            return std::abs(normal.x) > std::abs(normal.y) ? 
                (std::abs(normal.x) > std::abs(normal.z) ? 0 : 2) : 
                (std::abs(normal.y) > std::abs(normal.z) ? 1 : 2);
        }
    };

    struct STLMesh
    {   
        uint32_t numTriangles;
        std::vector<Triangle> triangleList;
        ~STLMesh() {
            triangleList.clear();
            triangleList.shrink_to_fit();
        }
    };

    struct VTKMesh
    {
        std::vector<Vector3d> points;
        std::vector<std::vector<int>> cells;
        ~VTKMesh() {
            points.clear();
            points.shrink_to_fit();
            cells.clear();
            cells.shrink_to_fit();
        }
    };

    class VoxelGrid
    {
    public:
        VoxelGrid();
        ~VoxelGrid();
        VoxelGrid(const STLMesh *stlmesh, const int numX, const int numY, const int numZ);
        void Update(const STLMesh *stlmesh);
        void OutputVTKFile(const std::string &outputfile);
    private:
        int m_numVoxels;
        int m_numX, m_numY, m_numZ;
        Vector3d m_minGrid, m_maxGrid;
        Vector3d m_voxelSize;
        std::vector<int> m_grid;
        // -1: empty, 0: surface, 1: inside
        void MarkSurfaceVoxels(const Vector3d &voxelCoord);
        void MarkInsideVoxels(const Vector3d &voxelCoord);
        void ComfirmSurfaceVoxels(const STLMesh *stlmesh);
        void ComfirmInsideVoxels(const STLMesh *stlmesh);
        void ClearFreeBodyVoxels();
        
    };

    class Voxelizer
    {
    public:
        Voxelizer();
        ~Voxelizer();
        void ReadSTLFile(const std::string& filename);
        void WriteVTKFile(const std::string& filename);
        void OutputVoxelModel(const std::string& filename, const int numX, const int numY, const int numZ);
        void WriteVoxelFile(const std::string& filename);
        void OutputSTLInformation();
    private:
        STLMesh *m_stlmesh;
        VTKMesh *m_vtkmesh;
        VoxelGrid *m_voxelgrid;
        void LoadVTKMesh();
        void Voxelization(const int numX, const int numY, const int numZ);

    };
}

#endif // __VOXELIZER_H__