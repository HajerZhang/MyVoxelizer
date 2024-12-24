#include <Voxelizer.h>

#include <iostream>

using namespace voxel;

void stlProcess(Voxelizer *voxelizer, const std::string& filename)
{
    voxelizer->ReadSTLFile(filename);
    voxelizer->OutputSTLInformation();

    int numX, numY, numZ;
    std::cout << "Please Enter the Number of Voxels in X, Y, Z: ";
    std::cin >> numX >> numY >> numZ;

    while(true){
        std::cout << "Please Choose the Operation: " << std::endl;
        std::cout << "1. Write VTK File" << std::endl;
        std::cout << "2. Output Voxel Model" << std::endl;
        std::cout << "3. Get VoxelGrid Set" << std::endl;
        std::cout << "4. Write Voxel File" << std::endl;
        std::cout << "0. Exit" << std::endl;

        int operation;
        std::cout << "Enter the Operation: ";
        std::cin >> operation;

        auto appendSuffix = [](const std::string& suffix) {
            std::cout << "Please Enter the Output File Name: ";
            std::string filename;
            std::cin >> filename;
            filename += suffix;
            return filename;
        };

        std::string outputfile;
        switch (operation)
        {
        case 1:
            outputfile = appendSuffix(".vtk");
            voxelizer->WriteVTKFile(outputfile);
            break;
        case 2:
            outputfile = appendSuffix(".vtk");
            voxelizer->OutputVoxelModel(outputfile, numX, numY, numZ);
            break;
        case 3:
            outputfile = appendSuffix(".vtk");
            voxelizer->GetVoxelGridSet(outputfile);
            break;
        case 4:
            outputfile = appendSuffix(".xml");
            voxelizer->WriteVoxelFile(outputfile);
        case 0:
            return;
        }
    }
}

int main(int argc, char** argv)
{   
    
    Voxelizer *voxelizer = new Voxelizer();
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    
    std::string filename = argv[1];
    int pointIndex = filename.find_last_of(".");
    std::string suffix = filename.substr(pointIndex + 1);
    if(suffix == "stl"){
        stlProcess(voxelizer, filename);
    } else if (suffix == "vtk") {
        // vtkProcess(voxelizer, filename);
    } else {
        std::cout << "Unsupported File Format" << std::endl;
    }


   delete[] voxelizer;

    return 0;
}