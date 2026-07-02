from stl import mesh
from collections import defaultdict
import pyvista as pv
import numpy as np

STL_FILE = "ohisje_ai_focus_assistant.stl"

def load_stl():

    model = mesh.Mesh.from_file(STL_FILE)
    return model


def print_model_info(model):

    print("STL MODEL INFO")
    print()
    print("Number of triangles:", len(model.vectors))
    print()

    for i, triangle in enumerate(model.vectors[:5]):

        print(f"Triangle {i + 1}:")
        print(triangle)
        print()


def calculate_bounding_box(model):

    all_vertices = model.vectors.reshape(-1, 3)
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)

    print("BOUNDING BOX")
    print()
    print("Min coordinates:", min_coords)
    print("Max coordinates:", max_coords)
    print()
    return min_coords, max_coords


def check_watertight(model):

    edge_count = defaultdict(int)

    def normalize_vertex(v):
        return tuple(np.round(v, 5))

    for triangle in model.vectors:

        v1 = normalize_vertex(triangle[0])
        v2 = normalize_vertex(triangle[1])
        v3 = normalize_vertex(triangle[2])

        edges = [
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v3])),
            tuple(sorted([v3, v1]))
        ]

        for edge in edges:
            edge_count[edge] += 1

    invalid_edges = []

    for edge, count in edge_count.items():

        if count != 2:
            invalid_edges.append((edge, count))

    print("WATER-TIGHT CHECK")
    print()

    if len(invalid_edges) == 0:
        print("Model IS watertight")
    else:
        print("Model is NOT watertight")
        print()

        for edge, count in invalid_edges[:10]:
            print(edge, "appears", count, "times")


def visualize_model():

    mesh_plot = pv.read(STL_FILE)
    plotter = pv.Plotter(window_size=[1000, 700])
    plotter.add_mesh(
        mesh_plot,
        color="lightblue",
        show_edges=True,
        edge_color="black",
        line_width=1
    )

    plotter.add_axes()
    plotter.add_title("STL Model Visualization")
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":

    model = load_stl()
    print_model_info(model)
    calculate_bounding_box(model)
    check_watertight(model)
    visualize_model()