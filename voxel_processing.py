from stl import mesh
import numpy as np
from collections import deque
import pyvista as pv

STL_FILE = "ohisje_ai_focus_assistant.stl"
voxel_size = 5


def load_model():
    return mesh.Mesh.from_file(STL_FILE)


def create_surface_voxels(model):
    all_vertices = model.vectors.reshape(-1, 3)

    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)

    grid_size = ((max_coords - min_coords) / voxel_size).astype(int) + 6
    surface_voxels = np.zeros(grid_size, dtype=bool)

    def point_to_voxel(point):
        return ((point - min_coords) / voxel_size).astype(int) + 3

    for triangle in model.vectors:
        v1, v2, v3 = triangle

        for a in np.linspace(0, 1, 80):
            for b in np.linspace(0, 1 - a, 80):
                c = 1 - a - b
                point = a * v1 + b * v2 + c * v3

                x, y, z = point_to_voxel(point)

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nx = x + dx
                            ny = y + dy
                            nz = z + dz

                            if (
                                0 <= nx < grid_size[0]
                                and 0 <= ny < grid_size[1]
                                and 0 <= nz < grid_size[2]
                            ):
                                surface_voxels[nx, ny, nz] = True

    return surface_voxels, min_coords, grid_size


def flood_fill_outside(surface_voxels, grid_size):
    outside = np.zeros(grid_size, dtype=bool)
    queue = deque([(0, 0, 0)])

    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    while queue:
        x, y, z = queue.popleft()

        if (
            x < 0 or x >= grid_size[0]
            or y < 0 or y >= grid_size[1]
            or z < 0 or z >= grid_size[2]
        ):
            continue

        if outside[x, y, z]:
            continue

        if surface_voxels[x, y, z]:
            continue

        outside[x, y, z] = True

        for dx, dy, dz in directions:
            queue.append((x + dx, y + dy, z + dz))

    return outside


def create_labels(surface_voxels, outside):
    labels = np.zeros(surface_voxels.shape, dtype=np.uint8)

    # 0 = air
    # 1 = surface/material voxel
    # 2 = outside air
    # 3 = outside shell
    # 4 = material/interior wall area

    labels[surface_voxels] = 1
    labels[outside] = 2

    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    # surface voxels touching outside air become outer shell
    sx, sy, sz = surface_voxels.shape

    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                if labels[x, y, z] == 1:
                    for dx, dy, dz in directions:
                        nx = x + dx
                        ny = y + dy
                        nz = z + dz

                        if (
                            0 <= nx < sx
                            and 0 <= ny < sy
                            and 0 <= nz < sz
                            and labels[nx, ny, nz] == 2
                        ):
                            labels[x, y, z] = 3
                            break

    # remaining non-outside empty space is treated as material/interior region
    labels[(labels == 0)] = 4

    return labels


def calculate_volume(labels):
    material_voxels = (
        (labels == 1)
        | (labels == 3)
        | (labels == 4)
    )

    material_count = np.sum(material_voxels)
    voxel_volume = voxel_size ** 3
    total_volume = material_count * voxel_volume

    print("=== VOLUME RESULT ===")
    print()
    print("Material voxels:", material_count)
    print("Voxel volume:", voxel_volume)
    print("Estimated material volume:", total_volume)

    return material_voxels, total_volume


def visualize_voxels(material_voxels, min_coords):
    points = []

    for x in range(material_voxels.shape[0]):
        for y in range(material_voxels.shape[1]):
            for z in range(material_voxels.shape[2]):
                if material_voxels[x, y, z]:
                    point = min_coords + np.array([x, y, z]) * voxel_size
                    points.append(point)

    if len(points) == 0:
        print("No material voxels found")
        return

    point_cloud = pv.PolyData(np.array(points))

    plotter = pv.Plotter()
    plotter.add_mesh(
        point_cloud,
        render_points_as_spheres=True,
        point_size=3,
        color="green"
    )

    plotter.add_axes()
    plotter.add_title("Voxelized Material Model")
    plotter.show()


if __name__ == "__main__":
    model = load_model()

    surface_voxels, min_coords, grid_size = create_surface_voxels(model)

    outside = flood_fill_outside(surface_voxels, grid_size)

    labels = create_labels(surface_voxels, outside)

    material_voxels, total_volume = calculate_volume(labels)

    visualize_voxels(material_voxels, min_coords)