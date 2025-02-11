﻿////////////////////////////////////////////////////////////////////////////////////////    
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
#include <sstream>

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
        Vector3d operator-(const Vector3d& other) const {
            return Vector3d(x - other.x, y - other.y, z - other.z);
        }

        Vector3d operator+(const Vector3d& other) const {
            return Vector3d(x + other.x, y + other.y, z + other.z);
        }

        Vector3d operator*(float scalar) const {
            return Vector3d(x * scalar, y * scalar, z * scalar);
        }
        
        Vector3d operator/(float scalar) const {
            return Vector3d(x / scalar, y / scalar, z / scalar);
        }

        float dot(const Vector3d& other) const {
            return x * other.x + y * other.y + z * other.z;
        }

        Vector3d cross(const Vector3d& other) const {
            return Vector3d(
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x
            );
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

    typedef enum{
        CELL_SET = 0,
        POINT_SET = 1
    } SetType;

    struct Set
    {
        SetType type;
        std::vector<int> index;
        ~Set() {
            index.clear();
            index.shrink_to_fit();
        }
    };

    typedef enum{
        SURFACE_IN = -1,
        SURFACE_OUT = 1,
        PENETRATE = 0
    } ChooseType;

    typedef enum{
        ADD = 1,
        SUB = -1,
    } InteractType;

    class VoxelGrid
    {
    public:
        VoxelGrid();
        ~VoxelGrid();
        VoxelGrid(const STLMesh *stlmesh, const int numX, const int numY, const int numZ);
        void Update(const STLMesh *stlmesh);
        void OutputVTKFile(const std::string &outputfile);
        void OutputPostVTKFile(const std::string &outputfile);
        void TwoPoint2GetSet();
        void OutputXMLFile(const std::string &outputfile);
    private:
        int m_numVoxels;
        int m_numX, m_numY, m_numZ;
        Vector3d m_minGrid, m_maxGrid;
        Vector3d m_voxelSize;
        int m_numSurfaceVoxels = 0;
        int m_numInsideVoxels = 0;
        std::vector<int> m_grid;
        // -1: empty, 0: surface, 1: inside
        std::vector<Set> m_setList;
        void MarkSurfaceVoxels(const Vector3d &voxelCoord);
        void MarkInsideVoxels(const Vector3d &voxelCoord);
        void ComfirmSurfaceVoxels(const STLMesh *stlmesh);
        void ComfirmInsideVoxels();
        void ClearFreeBodyVoxels();
        void GetCellSet
        (   
            const Vector3d &onePoint, const Vector3d &twoPoint, 
            const InteractType &interactType, std::vector<int> &index
        );
        void GetPointSet
        (   
            const Vector3d &onePoint, const Vector3d &twoPoint, 
            const ChooseType chooseType,const InteractType &interactType, 
            std::vector<int> &index
        );
    };


    class Voxelizer
    {
    public:
        Voxelizer();
        ~Voxelizer();
        void ReadSTLFile(const std::string& filename);
        void WriteVTKFile(const std::string& filename);
        void OutputVoxelModel(const std::string& filename, const int numX, const int numY, const int numZ);
        void GetVoxelGridSet(const std::string &outputfile);
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