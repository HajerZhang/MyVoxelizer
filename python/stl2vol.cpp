#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <Eigen/Dense>

// 体素网格定义
struct VoxelGrid {
    Eigen::Vector3d origin;  // 网格原点
    double voxel_size;       // 体素大小
    std::vector<std::vector<std::vector<bool>>> grid;  // 占用状态

    // 初始化网格
    VoxelGrid(const Eigen::Vector3d& origin, double voxel_size, int x_size, int y_size, int z_size)
        : origin(origin), voxel_size(voxel_size) {
        grid.resize(x_size, std::vector<std::vector<bool>>(y_size, std::vector<bool>(z_size, false)));
    }

    // 获取体素坐标
    Eigen::Vector3i getVoxelCoord(const Eigen::Vector3d& point) {
        return ((point - origin) / voxel_size).cast<int>();
    }

    // 标记体素为占用
    void markVoxel(const Eigen::Vector3i& voxel_coord) {
        if (voxel_coord.x() >= 0 && voxel_coord.x() < grid.size() &&
            voxel_coord.y() >= 0 && voxel_coord.y() < grid[0].size() &&
            voxel_coord.z() >= 0 && voxel_coord.z() < grid[0][0].size()) {
            grid[voxel_coord.x()][voxel_coord.y()][voxel_coord.z()] = true;
        }
    }
};

// STL 三角形定义
struct Triangle {
    Eigen::Vector3d v0, v1, v2;  // 三角形顶点
    Eigen::Vector3d normal;      // 法向量
};

// 解析二进制 STL 文件
std::vector<Triangle> parseBinarySTL(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    // 跳过80字节的文件头和4字节的三角形数量
    file.seekg(84);
    std::vector<Triangle> triangles;

    while (file) {
        Triangle tri;
        float normal[3], v0[3], v1[3], v2[3];
        file.read(reinterpret_cast<char*>(normal), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v0), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v1), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(v2), 3 * sizeof(float));
        file.ignore(2);  // 跳过属性字节

        tri.normal = Eigen::Vector3d(normal[0], normal[1], normal[2]);
        tri.v0 = Eigen::Vector3d(v0[0], v0[1], v0[2]);
        tri.v1 = Eigen::Vector3d(v1[0], v1[1], v1[2]);
        tri.v2 = Eigen::Vector3d(v2[0], v2[1], v2[2]);
        triangles.push_back(tri);
    }

    return triangles;
}

// 三角形与体素相交检测
bool triangleIntersectsVoxel(const Triangle& tri, const Eigen::Vector3d& voxel_min, double voxel_size) {
    // 简单的 AABB 交点检测
    Eigen::Vector3d voxel_max = voxel_min + Eigen::Vector3d::Constant(voxel_size);
    Eigen::AlignedBox3d voxel(voxel_min, voxel_max);
    Eigen::AlignedBox3d triangle_box(tri.v0, tri.v0);
    triangle_box.extend(tri.v1);
    triangle_box.extend(tri.v2);
    return voxel.intersects(triangle_box);
}

// 主体体素化函数
void voxelize(const std::vector<Triangle>& triangles, VoxelGrid& voxel_grid) {
    for (const auto& tri : triangles) {
        Eigen::AlignedBox3d bounding_box(tri.v0, tri.v0);
        bounding_box.extend(tri.v1);
        bounding_box.extend(tri.v2);

        // 获取包围盒的体素范围
        Eigen::Vector3i min_voxel = voxel_grid.getVoxelCoord(bounding_box.min());
        Eigen::Vector3i max_voxel = voxel_grid.getVoxelCoord(bounding_box.max());

        for (int x = min_voxel.x(); x <= max_voxel.x(); ++x) {
            for (int y = min_voxel.y(); y <= max_voxel.y(); ++y) {
                for (int z = min_voxel.z(); z <= max_voxel.z(); ++z) {
                    Eigen::Vector3d voxel_min = voxel_grid.origin + Eigen::Vector3d(x, y, z) * voxel_grid.voxel_size;
                    if (triangleIntersectsVoxel(tri, voxel_min, voxel_grid.voxel_size)) {
                        voxel_grid.markVoxel({x, y, z});
                    }
                }
            }
        }
    }
}

int main() {
    // 读取 STL 文件
    std::string stl_file = "sofa.stl";
    auto triangles = parseBinarySTL(stl_file);

    // 创建体素网格
    Eigen::Vector3d origin(0, 0, 0);
    double voxel_size = 0.1;  // 体素大小
    int grid_size = 8;      // 网格尺寸
    VoxelGrid voxel_grid(origin, voxel_size, grid_size, grid_size, grid_size);

    // 模型体素化
    voxelize(triangles, voxel_grid);

    // 输出体素化结果
    std::cout << "Voxelization complete!" << std::endl;
    return 0;
}