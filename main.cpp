#include <Voxelizer.cuh>

#include <iostream>

int main(int argc, char** argv)
{   
    using namespace voxel;
    Voxelizer *voxelizer = new Voxelizer();
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <stl file>" << std::endl;
        return 1;
    }

    voxelizer->ReadSTLFile(argv[1]);
    voxelizer->OutputSTLInformation();

    while(true){
        std::cout << "Please Choose the Operation: " << std::endl;
        std::cout << "1. Write VTK File" << std::endl;
        std::cout << "2. Output Voxel Model" << std::endl;
        std::cout << "3. Write Voxel Information" << std::endl;
        std::cout << "0. Exit" << std::endl;

        int operation;
        std::cout << "Enter the Operation: ";
        std::cin >> operation;

        auto appendSuffix = [](const std::string& suffix) {
            std::cout << "Please Enter the Output File Name: ";
            std::string filename;
            std::cin >> filename;
            filename += ".vtk";
            return filename;
        };

        int numX, numY, numZ;

        std::string outputfile;
        switch (operation)
        {
        case 1:
            outputfile = appendSuffix(".vtk");
            voxelizer->WriteVTKFile(outputfile);
            break;
        case 2:
            outputfile = appendSuffix(".vtk");
            std::cout << "Please Enter the Number of Voxels in X, Y, Z: ";
            std::cin >> numX >> numY >> numZ;
            voxelizer->OutputVoxelModel(outputfile, numX, numY, numZ);
            break;
        case 3:
            outputfile = appendSuffix(".txt");
            std::cout << "Please Enter the Number of Voxels in X, Y, Z: ";
            std::cin >> numX >> numY >> numZ;
            // voxelizer->WriteVoxelInformation("voxel.txt");
            break;
        case 0:
            delete voxelizer;
            return 0;
        }
    }

    return 0;
}