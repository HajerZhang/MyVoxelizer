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
#include <tinyxml2.h>

#include <cmath>
#include <algorithm>
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
}

VoxelGrid::~VoxelGrid()
{
    m_grid.clear();
    // m_grid.shrink_to_fit();
    m_setList.clear();
    // m_setList.shrink_to_fit();
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

    # pragma omp parallel
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

bool isSegmentIntersectTriangle(const Vector3d& p1, const Vector3d& p2, const Triangle& triangle)
{
    Vector3d edge1 = triangle.v1 - triangle.v0;
    Vector3d edge2 = triangle.v2 - triangle.v0;
    Vector3d dir = p2 - p1;
    Vector3d h = dir.cross(edge2);

    double a = edge1.dot(h);
    if (std::abs(a) < 1e-6)
        return false;

    double f = 1.0 / a;
    Vector3d s = p1 - triangle.v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
        return false;

    Vector3d q = s.cross(edge1);
    double v = f * dir.dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double t = f * edge2.dot(q);
    return t >= 0.0 && t <= 1.0;
}

bool voxelIntersectsTriangle(const Vector3d& voxelMin, const Vector3d& voxelMax,
                              const Triangle& triangle) {
    Vector3d vertices[8] = {
        Vector3d(voxelMin.x, voxelMin.y, voxelMin.z),
        Vector3d(voxelMax.x, voxelMin.y, voxelMin.z),
        Vector3d(voxelMin.x, voxelMax.y, voxelMin.z),
        Vector3d(voxelMax.x, voxelMax.y, voxelMin.z),
        Vector3d(voxelMin.x, voxelMin.y, voxelMax.z),
        Vector3d(voxelMax.x, voxelMin.y, voxelMax.z),
        Vector3d(voxelMin.x, voxelMax.y, voxelMax.z),
        Vector3d(voxelMax.x, voxelMax.y, voxelMax.z)
    };

    int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0}, 
        {4, 5}, {5, 7}, {7, 6}, {6, 4}, 
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  
    };

    for (int i = 0; i < 12; i++) {
        Vector3d p1 = vertices[edges[i][0]];
        Vector3d p2 = vertices[edges[i][1]];
        if (isSegmentIntersectTriangle(p1, p2, triangle)) {
            return true;
        }
    }

    return false;
}

bool isPointInTriangle(const Vector3d& point, const Vector3d& v0, const Vector3d& v1, const Vector3d& v2)
{
    Vector3d v0v1 = v1 - v0;
    Vector3d v0v2 = v2 - v0;
    Vector3d vp = point - v0;

    double dot00 = v0v1.dot(v0v1);
    double dot01 = v0v1.dot(v0v2);
    double dot02 = v0v1.dot(vp);
    double dot11 = v0v2.dot(v0v2);
    double dot12 = v0v2.dot(vp);

    double inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    if (u < 0 || u > 1) // if u out of range, return directly
    {
        return false;
    }

    double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
    if (v < 0 || v > 1) // if v out of range, return directly
    {
        return false;
    }

    return u + v <= 1;
}

double Point2TriangleDistance(const Vector3d& point, const Triangle& triangle, Vector3d &keyPoint)
{
    Vector3d v0 = point - triangle.v0;
    Vector3d normal = triangle.normal;
    double lenghNormal = sqrt(normal.dot(normal));
    normal = triangle.normal/lenghNormal;
    double distance2Plane = v0.dot(normal);

    Vector3d proj = point -  normal * distance2Plane;
    if(isPointInTriangle(proj, triangle.v0, triangle.v1, triangle.v2))
    {   
        keyPoint = proj;
        return distance2Plane;
    }else{
        Vector3d edge10 = triangle.v1 - triangle.v0;
        Vector3d edge20 = triangle.v2 - triangle.v0;
        Vector3d edge21 = triangle.v2 - triangle.v1;

        double length10 = sqrt(edge10.dot(edge10));
        double length20 = sqrt(edge20.dot(edge20));
        double length21 = sqrt(edge21.dot(edge21));

        Vector3d edge10Point = triangle.v0 + edge10 * (proj - triangle.v0).dot(edge10) / length10;
        Vector3d edge20Point = triangle.v0 + edge20 * (proj - triangle.v0).dot(edge20) / length20;
        Vector3d edge21Point = triangle.v1 + edge21 * (proj - triangle.v1).dot(edge21) / length21;

        double distance10 = (proj - edge10Point).dot(proj - edge10Point);
        double distance20 = (proj - edge20Point).dot(proj - edge20Point);
        double distance21 = (proj - edge21Point).dot(proj - edge21Point);

        keyPoint = distance10 < distance20 ? (distance10 < distance21 ? edge10Point : edge21Point) : (distance20 < distance21 ? edge20Point : edge21Point);
        return sqrt(std::min(distance10, std::min(distance20, distance21)) + distance2Plane * distance2Plane);    
    }
}


bool distanceCheck(Vector3d& minVoxel, const Vector3d& maxVoxel, const Triangle& triangle)
{
    Vector3d centerVoxel = Vector3d(
        (minVoxel.x + maxVoxel.x) / 2.0,
        (minVoxel.y + maxVoxel.y) / 2.0,
        (minVoxel.z + maxVoxel.z) / 2.0
    );
    double voxelLength = 0.5 * sqrt(
        (maxVoxel.x - minVoxel.x) * (maxVoxel.x - minVoxel.x) +
        (maxVoxel.y - minVoxel.y) * (maxVoxel.y - minVoxel.y) +
        (maxVoxel.z - minVoxel.z) * (maxVoxel.z - minVoxel.z)
    );

    Vector3d keyPoint;
    double distance = Point2TriangleDistance(centerVoxel, triangle, keyPoint);

    return distance < voxelLength;
}

void VoxelGrid::ComfirmSurfaceVoxels(const STLMesh *stlmesh)
{   
    int numSurfaceVoxels = 0;
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



                    if(
                        // #define AABB
                        #ifdef AABB
                            aabbCheck(minTri, maxTri, minVoxel, maxVoxel)
                        #endif
                        #define INTERSECT
                        #ifdef INTERSECT
                            voxelIntersectsTriangle(minVoxel, maxVoxel, triangle)
                        #endif
                        // #define DISTANCE
                        #ifdef DISTANCE
                            distanceCheck(minVoxel, maxVoxel, triangle)
                        #endif
                    )
                    {
                        MarkSurfaceVoxels(Vector3d(x, y, z));
                        # pragma omp atomic
                        numSurfaceVoxels++;
                        std::cout << "\r" << numSurfaceVoxels << " Surface Voxels have been marked           " << std::flush;
                    }

                }
            }
        }
    }

    m_numSurfaceVoxels = numSurfaceVoxels;
}

void VoxelGrid::ComfirmInsideVoxels()
{   
    int numInsideVoxels = 0;
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

                if(inside) {
                    MarkInsideVoxels(voxelCoord);
                    # pragma omp atomic
                    numInsideVoxels++;
                    std::cout << "\r" <<  numInsideVoxels << " Inside Voxels have been marked                " << std::flush;
                }
                
            }
        }
    }

    m_numInsideVoxels = numInsideVoxels;
}

void VoxelGrid::Update(const STLMesh *stlmesh)
{
    ComfirmSurfaceVoxels(stlmesh);
    ComfirmInsideVoxels();
    std::cout << "Voxelization has been finished" << std::endl;
    std::cout << "Finally found:" << std::endl;
    std::cout << m_numSurfaceVoxels << " Surface Voxels" << std::endl;
    std::cout << m_numInsideVoxels << " Inside Voxels" << std::endl;
    std::cout << "in total " << m_numVoxels << " Voxels" << std::endl;
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



void VoxelGrid::TwoPoint2GetSet()
{   
    const Vector3d minGrid = m_minGrid;
    const Vector3d maxGrid = m_maxGrid;
    const int numX = m_numX;
    const int numY = m_numY;
    const int numZ = m_numZ;
    std::cout << "The range of Voxel Grid is: " << std::endl;
    std::cout << "X: " << minGrid.x << " - " << maxGrid.x << std::endl;
    std::cout << "Y: " << minGrid.y << " - " << maxGrid.y << std::endl;
    std::cout << "Z: " << minGrid.z << " - " << maxGrid.z << std::endl;
    std::cout << "The number of Voxels in X, Y, Z is: " << std::endl;
    std::cout << numX << ", " << numY << ", " << numZ << std::endl;

    while(true){
        std::cout << "Please Choose the Operation: " << std::endl;
        std::cout << "1. To Get Element Set" << std::endl;
        std::cout << "2. To Get Point Set" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Enter the Operation: ";
        int operation;
        std::cin >> operation;

        if(operation == 0) break;

        auto get2Point = [](Vector3d &OnePoint, Vector3d &TwoPoint){
            std::cout << "Please Enter the First Point: ";
            std::cin >> OnePoint.x >> OnePoint.y >> OnePoint.z;
            std::cout << "Please Enter the Second Point: ";
            std::cin >> TwoPoint.x >> TwoPoint.y >> TwoPoint.z;
        };

        auto selectMode = [](ChooseType &chooseType){
            std::cout << "Please Choose the Mode: " << std::endl;
            std::cout << "1. Surface In; " ;
            std::cout << "2. Surface Out; ";
            std::cout << "3. Penetrate" << std::endl;
            std::cout << "Enter the Mode: ";
            int mode;
            std::cin >> mode;
            if(mode == 1) chooseType = SURFACE_IN;
            if(mode == 2) chooseType = SURFACE_OUT;
            if(mode == 3) chooseType = PENETRATE;
        };

        auto setOpera = [](InteractType &interactType){
            std::cout << "Please Choose the Operation: " << std::endl;
            std::cout << "1. Add; " ;
            std::cout << "2. Sub; ";
            std::cout << "Enter the Operation: ";
            int mode;
            std::cin >> mode;
            if(mode == 1) interactType = ADD;
            if(mode == 2) interactType = SUB;
        };


        Set setBuffer;
        Vector3d OnePoint, TwoPoint;
        ChooseType chooseType;
        InteractType interactType;
        if(operation == 1){
            setBuffer.type = CELL_SET;
            while(true){
                int flag;
                std::cout << "There are " << setBuffer.index.size() << " elements in the set" << std::endl;
                std::cout << "You wanna add more elements or exit(1/0): ";
                std::cin >> flag;
                if(flag == 0) break;
                get2Point(OnePoint, TwoPoint);
                setOpera(interactType);
                GetCellSet(OnePoint, TwoPoint, interactType, setBuffer.index);
            }
        }
        if(operation == 2){
            setBuffer.type = POINT_SET;
            while(true){
                int flag;
                std::cout << "There are " << setBuffer.index.size() << " points in the set" << std::endl;
                std::cout << "You wanna add more elements or exit(1/0): ";
                std::cin >> flag;
                if(flag == 0) break;
                get2Point(OnePoint, TwoPoint);
                selectMode(chooseType);
                setOpera(interactType);
                GetPointSet(OnePoint, TwoPoint, chooseType, interactType, setBuffer.index);
            }
        }
        m_setList.push_back(setBuffer);
    }
}

void VoxelGrid::GetCellSet
(
    const Vector3d &onePoint, const Vector3d &twoPoint, 
    const InteractType &interactType, std::vector<int> &index
)
{
    const Vector3d minGrid = m_minGrid;
    const Vector3d maxGrid = m_maxGrid;
    const int numX = m_numX;
    const int numY = m_numY;
    const int numZ = m_numZ;

    Vector3d minPoint = Vector3d(
        std::min(onePoint.x, twoPoint.x),
        std::min(onePoint.y, twoPoint.y),
        std::min(onePoint.z, twoPoint.z)
    );
    Vector3d maxPoint = Vector3d(
        std::max(onePoint.x, twoPoint.x),
        std::max(onePoint.y, twoPoint.y),
        std::max(onePoint.z, twoPoint.z)
    );

    Vector3d minVoxel = Vector3d(
        std::max(0, (int)std::floor((minPoint.x - minGrid.x) / m_voxelSize.x)),
        std::max(0, (int)std::floor((minPoint.y - minGrid.y) / m_voxelSize.y)),
        std::max(0, (int)std::floor((minPoint.z - minGrid.z) / m_voxelSize.z))
    );

    Vector3d maxVoxel = Vector3d(
        std::min(numX - 1, (int)std::ceil((maxPoint.x - minGrid.x) / m_voxelSize.x)),
        std::min(numY - 1, (int)std::ceil((maxPoint.y - minGrid.y) / m_voxelSize.y)),
        std::min(numZ - 1, (int)std::ceil((maxPoint.z - minGrid.z) / m_voxelSize.z))
    );

    for(int z = minVoxel.z; z <= maxVoxel.z; z++){
        for(int y = minVoxel.y; y <= maxVoxel.y; y++){
            for(int x = minVoxel.x; x <= maxVoxel.x; x++){
                Vector3d voxelCoord = Vector3d(x, y, z);
                if(m_grid[z * numX * numY + y * numX + x] == -1) continue;
                if(interactType == ADD) index.push_back(z * numX * numY + y * numX + x);
                if(interactType == SUB) index.erase(std::remove(index.begin(), index.end(), z * numX * numY + y * numX + x), index.end());
            }
        }
    }
}

void VoxelGrid::GetPointSet
(
    const Vector3d &onePoint, const Vector3d &twoPoint, 
    const ChooseType chooseType,const InteractType &interactType, 
    std::vector<int> &index
)
{
    const Vector3d minGrid = m_minGrid;
    const Vector3d maxGrid = m_maxGrid;
    const int numX = m_numX;
    const int numY = m_numY;
    const int numZ = m_numZ;

    Vector3d minPoint = Vector3d(
        std::min(onePoint.x, twoPoint.x),
        std::min(onePoint.y, twoPoint.y),
        std::min(onePoint.z, twoPoint.z)
    );
    Vector3d maxPoint = Vector3d(
        std::max(onePoint.x, twoPoint.x),
        std::max(onePoint.y, twoPoint.y),
        std::max(onePoint.z, twoPoint.z)
    );

    minPoint = Vector3d(
        std::max(0, (int)std::ceil((minPoint.x - minGrid.x) / m_voxelSize.x)),
        std::max(0, (int)std::ceil((minPoint.y - minGrid.y) / m_voxelSize.y)),
        std::max(0, (int)std::ceil((minPoint.z - minGrid.z) / m_voxelSize.z))
    );

    maxPoint = Vector3d(
        std::min(numX, (int)std::floor((maxPoint.x - minGrid.x) / m_voxelSize.x)),
        std::min(numY, (int)std::floor((maxPoint.y - minGrid.y) / m_voxelSize.y)),
        std::min(numZ, (int)std::floor((maxPoint.z - minGrid.z) / m_voxelSize.z))
    );

    for(int z = minPoint.z; z <= maxPoint.z; z++){
        for(int y = minPoint.y; y <= maxPoint.y; y++){
            for(int x = minPoint.x; x <= maxPoint.x; x++){
                Vector3d pointCoord = Vector3d(x, y, z);
                std::vector<int> voxelType;
                for(int ei = -1; ei < 1; ei++){
                    for(int ej = -1; ej < 1; ej++){
                        for(int ek = -1; ek < 1; ek++){
                            if(x + ei < 0 || x + ei > numX) continue;
                            if(y + ej < 0 || y + ej > numY) continue;
                            if(z + ek < 0 || z + ek > numZ) continue;
                            voxelType.push_back(m_grid[(z + ek) * numX * numY + (y + ej) * numX + (x + ei)]);
                        }
                    }
                }
                std::sort(voxelType.begin(), voxelType.end());
                voxelType.erase(std::unique(voxelType.begin(), voxelType.end()), voxelType.end());

                if(chooseType == SURFACE_IN){
                    if(voxelType.size() == 2 && voxelType[0] == 0 && voxelType[1] == 1){
                        const int pointIndex = z * (numX + 1) * (numY + 1) + y * (numX + 1) + x;
                        if(interactType == ADD) index.push_back(pointIndex);
                        if(interactType == SUB) index.erase(std::remove(index.begin(), index.end(), pointIndex), index.end());
                    }
                }
                if(chooseType == SURFACE_OUT){
                    if(voxelType.size() == 2 && voxelType[0] == -1 && voxelType[1] == 0){
                        const int pointIndex = z * (numX + 1) * (numY + 1) + y * (numX + 1) + x;
                        if(interactType == ADD) index.push_back(pointIndex);
                        if(interactType == SUB) index.erase(std::remove(index.begin(), index.end(), pointIndex), index.end());
                    }
                }
                if(chooseType == PENETRATE){
                    if(voxelType.size() == 1 && voxelType[0] == -1){
                    }else{
                        const int pointIndex = z * (numX + 1) * (numY + 1) + y * (numX + 1) + x;
                        if(interactType == ADD) index.push_back(pointIndex);
                        if(interactType == SUB) index.erase(std::remove(index.begin(), index.end(), pointIndex), index.end());
                    }
                }
                voxelType.clear();
                voxelType.shrink_to_fit();
            }
        }
    }

}

void VoxelGrid::OutputPostVTKFile(const std::string &outputfile)
{
    std::ofstream file(outputfile);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    std::vector<int> cell_data = m_grid;
    std::vector<int> point_data((m_numX+1)*(m_numY+1)*(m_numZ+1), 0);

    for(int i = 0; i < m_setList.size(); i++){
        if(m_setList[i].type == CELL_SET){
            int setId = i + 2;
            for(auto index : m_setList[i].index){
                if(index < 0 || index >= m_numVoxels) continue;
                cell_data[index] = setId;
            }
        }
        if(m_setList[i].type == POINT_SET){
            int setId = i + 1;
            for(auto index : m_setList[i].index){
                if(index < 0 || index >= (m_numX+1)*(m_numY+1)*(m_numZ+1)) continue;
                point_data[index] = setId;
            }
        }
    }

    file << "# vtk DataFile Version 3.0" << std::endl;
    file << "Voxel Grid" << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET STRUCTURED_POINTS" << std::endl;
    file << "DIMENSIONS " << m_numX + 1 << " " << m_numY + 1 << " " << m_numZ + 1 << std::endl;
    file << "SPACING " << m_voxelSize.x << " " << m_voxelSize.y << " " << m_voxelSize.z << std::endl;
    file << "ORIGIN " << m_minGrid.x << " " << m_minGrid.y << " " << m_minGrid.z << std::endl;

    file << "POINT_DATA " << (m_numX + 1) * (m_numY + 1) * (m_numZ + 1) << std::endl;
    file << "SCALARS point_type double 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < (m_numX + 1) * (m_numY + 1) * (m_numZ + 1); i++)
    {
        file << double(point_data[i]) << std::endl;
    }
    file << "CELL_DATA " << m_numVoxels << std::endl;
    file << "SCALARS cell_type double" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < m_numVoxels; i++)
    {
        file << double(cell_data[i]) << std::endl;
    }

    file.close();

    cell_data.clear();
    cell_data.shrink_to_fit();

    point_data.clear();
    point_data.shrink_to_fit();
}

void VoxelGrid::OutputXMLFile(const std::string &outputfile) {
    using namespace tinyxml2;
    XMLDocument doc;

    // Root element
    auto *root = doc.NewElement("VoxelStructure");
    doc.InsertFirstChild(root);

    // Grid dimensions
    auto *dimensions = doc.NewElement("Dimensions");
    dimensions->SetAttribute("numX", m_numX);
    dimensions->SetAttribute("numY", m_numY);
    dimensions->SetAttribute("numZ", m_numZ);
    root->InsertEndChild(dimensions);

    // Grid Orgins
    auto *orgins = doc.NewElement("Origins");
    orgins->SetAttribute("MinX", m_minGrid.x);
    orgins->SetAttribute("MinY", m_minGrid.y);
    orgins->SetAttribute("MinZ", m_minGrid.z);
    root->InsertEndChild(orgins);

    // Grid Bounds
    auto *bounds = doc.NewElement("Bounds");
    bounds->SetAttribute("MaxX", m_maxGrid.x);
    bounds->SetAttribute("MaxY", m_maxGrid.y);
    bounds->SetAttribute("MaxZ", m_maxGrid.z);
    root->InsertEndChild(bounds);

    // Voxel size
    auto *voxelSize = doc.NewElement("VoxelSize");
    voxelSize->SetAttribute("sizeX", m_voxelSize.x);
    voxelSize->SetAttribute("sizeY", m_voxelSize.y);
    voxelSize->SetAttribute("sizeZ", m_voxelSize.z);
    root->InsertEndChild(voxelSize);

    // Grid data
    std::vector<int> sufaceIndex;
    std::vector<int> insideIndex;
    for(int i = 0; i < m_numVoxels; i++){
        if(m_grid[i] == 0) sufaceIndex.push_back(i);
        if(m_grid[i] == 1) insideIndex.push_back(i);
    }
    auto *gridData = doc.NewElement("GridData");
    auto *surface = doc.NewElement("SurfaceVoxels");
    surface->SetAttribute("num", sufaceIndex.size());
    {
        std::ostringstream oss;
        std::string indent = "\t\t\t";
        oss << "\n" << indent;
        for (size_t i = 0; i < sufaceIndex.size(); ++i) {
            oss << sufaceIndex[i];
            if ((i + 1) % 8 == 0) {
                oss << "\n" << indent;
            } else if (i != sufaceIndex.size() - 1) {
                oss << ",";
            }
        }
        surface->SetText(oss.str().c_str()); 
    }
    gridData->InsertEndChild(surface);

    auto *inside = doc.NewElement("InsideVoxels");
    inside->SetAttribute("num", insideIndex.size());
    {
        std::ostringstream oss;
        std::string indent = "\t\t\t";
        oss << "\n" << indent;
        for (size_t i = 0; i < sufaceIndex.size(); ++i) {
            oss << sufaceIndex[i];
            if ((i + 1) % 8 == 0) {
                oss << "\n" << indent;
            } else if (i != sufaceIndex.size() - 1) {
                oss << ",";
            }
        }
        inside->SetText(oss.str().c_str());
    }
    gridData->InsertEndChild(inside);
    root->InsertEndChild(gridData);
    sufaceIndex.clear();
    sufaceIndex.shrink_to_fit();
    insideIndex.clear();
    insideIndex.shrink_to_fit();

    // Sets
    auto *sets = doc.NewElement("Sets");
    for (int i = 0; i < m_setList.size(); i++) {
        const auto &set = m_setList[i];
        auto *setElement = doc.NewElement("Set");
        setElement->SetAttribute("type", set.type == CELL_SET ? "CELL_SET" : "POINT_SET");
        setElement->SetAttribute("id", i);
        setElement->SetAttribute("num", set.index.size());

        auto *indices = doc.NewElement("Indices");
        std::ostringstream oss;
        std::string indent = "\t\t\t";
        oss << "\n" << indent;
        for (size_t j = 0; j < set.index.size(); ++j) {
            oss << set.index[j];
            if ((j + 1) % 8 == 0) {
                oss << "\n" << indent;
            } else if (j != set.index.size() - 1) {
                oss << ",";
            }
        }
        indices->SetText(oss.str().c_str());
        setElement->InsertEndChild(indices);

        sets->InsertEndChild(setElement);
    }
    root->InsertEndChild(sets);

    // Save to file
    if (doc.SaveFile(outputfile.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Error: Failed to save XML file." << std::endl;
    } else {
        std::cout << "XML file saved to " << outputfile << std::endl;
    }
}

