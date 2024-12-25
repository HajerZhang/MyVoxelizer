import numpy as np
import heapq
from vtk import vtkPoints, vtkCellArray, vtkPolyData, vtkPolyDataWriter, vtkLine

def read_input_file(filename):
    nodes = []
    elements = []
    reading_nodes = True
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Element"):
                reading_nodes = False
                continue
            if reading_nodes:
                if line.startswith("Node"):  # Skip the header line
                    continue
                parts = line.split(',')
                node_id = int(parts[0].strip())  # Read node ID
                coords = list(map(float, parts[1:]))
                nodes.append(coords)
            else:
                parts = line.split(',')
                element_id = int(parts[0].strip())  # Read element ID
                node_ids = list(map(int, parts[1:]))
                elements.append(node_ids)
    
    return np.array(nodes), elements


def create_surface_vtk(nodes, elements, output_filename):
    vtk_points = vtkPoints()
    for node in nodes:
        vtk_points.InsertNextPoint(node)

    vtk_cells = vtkCellArray()
    for element in elements:
        vtk_cells.InsertNextCell(len(element), [node_id - 1 for node_id in element])

    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)

    writer = vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()

if __name__ == "__main__":
    input_filename = "input.txt"
    nodes, elements = read_input_file(input_filename)

    surface_output_filename = "output.vtk"
    create_surface_vtk(nodes, elements, surface_output_filename)
    print(f"surface mesh written to {surface_output_filename}")
