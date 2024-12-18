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
#include <omp.h>
#include <chrono>

using namespace voxel;

#ifndef OMP_THREADS
    #define OMP_THREADS 16
#endif

Voxelizer::Voxelizer()
{
    omp_set_num_threads(OMP_THREADS);
    m_stlmesh = new STLMesh();
}

Voxelizer::~Voxelizer()
{   
    if(m_stlmesh != nullptr) delete m_stlmesh;
    if(m_vtkmesh != nullptr) delete m_vtkmesh;
    if(m_voxelgrid != nullptr) delete m_voxelgrid;
}

void Voxelizer::ReadSTLFile(const std::string& filename)
{   
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    file.seekg(80);
    uint32_t num_triangles;
    file.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));
    m_stlmesh->numTriangles = num_triangles;
    m_stlmesh->triangleList.reserve(num_triangles);

    for(uint32_t i = 0; i < num_triangles; ++i) {
        Triangle tri;
        float normal[3], v0[3], v1[3], v2[3];
        file.read(reinterpret_cast<char*>(normal), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v0), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v1), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v2), 3 * sizeof(float));
        file.ignore(2);

        tri.normal = Vector3d(normal[0], normal[1], normal[2]);
        tri.v0 = Vector3d(v0[0], v0[1], v0[2]);
        tri.v1 = Vector3d(v1[0], v1[1], v1[2]);
        tri.v2 = Vector3d(v2[0], v2[1], v2[2]);
        m_stlmesh->triangleList.push_back(tri);
    }

    std::cout << "Already Read " << num_triangles << " triangles from " << filename << std::endl;
}

void Voxelizer::LoadVTKMesh()
{   
    if(m_stlmesh->triangleList.empty()){
        throw std::runtime_error("STL file not loaded");
        exit(1);
    }

    m_vtkmesh = new VTKMesh();
    
    m_vtkmesh->points.clear();
    m_vtkmesh->cells.clear();

    for(const auto& triangle : m_stlmesh->triangleList)
    {
        m_vtkmesh->points.push_back(triangle.v0);
        m_vtkmesh->points.push_back(triangle.v1);
        m_vtkmesh->points.push_back(triangle.v2);

        m_vtkmesh->cells.push_back({
            static_cast<int>(m_vtkmesh->points.size()) - 3,
            static_cast<int>(m_vtkmesh->points.size()) - 2,
            static_cast<int>(m_vtkmesh->points.size()) - 1
        });
    }

    std::cout << "Already Load " << m_vtkmesh->points.size() << " points and " << m_vtkmesh->cells.size() << " cells" << std::endl;

}

void Voxelizer::WriteVTKFile(const std::string &outputfile)
{   
    if(m_stlmesh->triangleList.empty()){
        throw std::runtime_error("STL file not loaded");
        exit(1);
    }

    std::ofstream file(outputfile);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    if(m_vtkmesh == nullptr) LoadVTKMesh();

    file << "# vtk DataFile Version 3.0\n";
    file << "VTK from Voxelizer\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    file << "POINTS " << m_vtkmesh->points.size() << " double\n";
    for(const auto& point : m_vtkmesh->points)
    {
        file << point.x << " " << point.y << " " << point.z << std::endl;
    }
    file << "POLYGONS " << m_vtkmesh->cells.size() << " " << 4 * m_vtkmesh->cells.size() << "\n";
    for (const auto& cell : m_vtkmesh->cells) {
        file << "3 " << cell[0] << " " << cell[1] << " " << cell[2] << "\n";
    }
    std::cout << "Already Write " << m_vtkmesh->points.size() << " points and " << m_vtkmesh->cells.size() << " cells to " << outputfile << std::endl;

    file.close();
}

void Voxelizer::Voxelization(const int numX, const int numY, const int numZ)
{
    m_voxelgrid = new VoxelGrid(m_stlmesh, numX, numY, numZ);
    auto start = std::chrono::high_resolution_clock::now();
    m_voxelgrid->Update(m_stlmesh);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Voxelization Cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

void Voxelizer::OutputVoxelModel(
    const std::string &outputfile,
    const int numX, const int numY, const int numZ
)
{   
    if(m_stlmesh->triangleList.empty()){
        throw std::runtime_error("STL file not loaded");
        exit(1);
    }

    if(m_voxelgrid == NULL) Voxelization(numX, numY, numZ);

    m_voxelgrid->OutputVTKFile(outputfile);

    std::cout << "Already Output Voxel Model to " << outputfile << std::endl;
}


void Voxelizer::OutputSTLInformation()
{
    if(m_stlmesh->triangleList.empty()){
        throw std::runtime_error("STL file not loaded");
        exit(1);
    }

    std::cout << "STL File Information: " << std::endl;
    std::cout << "Number of Triangles: " << m_stlmesh->numTriangles << std::endl;
    
    Vector3d minGrid = Vector3d(m_stlmesh->triangleList[0].v0.x, m_stlmesh->triangleList[0].v0.y, m_stlmesh->triangleList[0].v0.z);
    Vector3d maxGrid = Vector3d(m_stlmesh->triangleList[0].v0.x, m_stlmesh->triangleList[0].v0.y, m_stlmesh->triangleList[0].v0.z);

    for(auto triangle : m_stlmesh->triangleList)
    {
        if(triangle.v0.x < minGrid.x) minGrid.x = triangle.v0.x;
        if(triangle.v0.y < minGrid.y) minGrid.y = triangle.v0.y;
        if(triangle.v0.z < minGrid.z) minGrid.z = triangle.v0.z;

        if(triangle.v0.x > maxGrid.x) maxGrid.x = triangle.v0.x;
        if(triangle.v0.y > maxGrid.y) maxGrid.y = triangle.v0.y;
        if(triangle.v0.z > maxGrid.z) maxGrid.z = triangle.v0.z;
    }

    std::cout << "X Range: " << minGrid.x << " - " << maxGrid.x << std::endl;
    std::cout << "Y Range: " << minGrid.y << " - " << maxGrid.y << std::endl;
    std::cout << "Z Range: " << minGrid.z << " - " << maxGrid.z << std::endl;
}