import numpy as np
from stl import mesh

def stl_to_txt(stl_filename, txt_filename):
    stl_mesh = mesh.Mesh.from_file(stl_filename)

    # get all unique nodes
    nodes = {}
    for triangle in stl_mesh.vectors:
        for vertex in triangle:
            node_tuple = tuple(np.round(vertex, decimals=6))  # set precision to 6 decimal places
            if node_tuple not in nodes:
                nodes[node_tuple] = len(nodes) + 1  # 1-based index

    # create node and element lists
    node_list = list(nodes.keys())
    element_list = []

    for triangle in stl_mesh.vectors:
        element = [nodes[tuple(np.round(vertex, decimals=6))] for vertex in triangle]
        element_list.append(element)

    with open(txt_filename, 'w') as file:
        file.write("Node ID, X, Y, Z\n")
        for i, node in enumerate(node_list, start=1):
            file.write(f"{i}, {node[0]}, {node[1]}, {node[2]}\n")

        file.write("Element ID, Node 1, Node 2, Node 3\n")
        for i, element in enumerate(element_list, start=1):
            file.write(f"{i}, {element[0]}, {element[1]}, {element[2]}\n")

if __name__ == "__main__":
    stl_filename = "sofa.stl" 
    txt_filename = "input.txt"
    stl_to_txt(stl_filename, txt_filename)
    print(f"已将 {stl_filename} 转换为 {txt_filename}")