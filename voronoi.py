import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, MultiPolygon # Import MultiPolygon explicitly
# from shapely.ops import polygonize, unary_union
import random
import colorsys
from collections import defaultdict

# generate_distinct_colors function remains the same
def generate_distinct_colors(n):
    """Generates N visually distinct colors."""
    hsv_tuples = [(x * 1.0 / n, 0.8, 0.8) for x in range(n)]
    rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    random.shuffle(rgb_tuples)
    return rgb_tuples

def create_clustered_circular_voronoi(data, circle_radius=1.0, circle_center=(0, 0), label_fontsize=6):
    """
    Generates a clustered circular Voronoi diagram with labels.

    Args:
        data (list): List of dictionaries with 'name', 'weight', 'category'.
        circle_radius (float): Radius of the clipping circle.
        circle_center (tuple): Center (x, y) of the circle.
        label_fontsize (int): Font size for the name labels.

    Returns:
        tuple: (fig, ax) handles for the matplotlib plot.
    """
    if not data:
        print("Input data cannot be empty.")
        return None, None

    num_points = len(data)
    categories = sorted(list(set(item['category'] for item in data)))
    num_categories = len(categories)

    # --- 1. Calculate Category Weights and Angular Sectors ---
    # (This part remains the same as the previous version)
    category_weights = defaultdict(float)
    category_counts = defaultdict(int)
    total_weight = 0.0
    for item in data:
        weight = item.get('weight', 1.0)
        if weight < 0: weight = 0.0
        category_weights[item['category']] += weight
        category_counts[item['category']] += 1
        total_weight += weight

    if total_weight <= 0:
        print("Total weight is zero or negative. Falling back to equal sectors by count.")
        total_angle_alloc = sum(category_counts.values())
        if total_angle_alloc > 0:
            category_angles = {}
            current_angle = 0
            for category in categories:
                proportion = category_counts[category] / total_angle_alloc
                angle_size = proportion * 2 * np.pi
                category_angles[category] = (current_angle, current_angle + angle_size)
                current_angle += angle_size
        else:
            print("Error: No data points found.")
            return None, None
    else:
        category_angles = {}
        current_angle = 0.0
        for category in categories:
            proportion = category_weights[category] / total_weight
            angle_size = proportion * 2 * np.pi
            category_angles[category] = (current_angle, current_angle + angle_size)
            current_angle += angle_size

    # --- 2. Generate Seed Points within Category Sectors ---
    # (This part remains the same)
    np.random.seed(42)
    generated_points = []
    point_data_mapping = []

    for item in data:
        category = item['category']
        angle_start, angle_end = category_angles[category]
        r = np.sqrt(np.random.rand()) * circle_radius
        theta = angle_start + np.random.rand() * (angle_end - angle_start)
        x = circle_center[0] + r * np.cos(theta)
        y = circle_center[1] + r * np.sin(theta)
        generated_points.append((x, y))
        point_data_mapping.append(item)

    points = np.array(generated_points)

    # --- Add Boundary Points ---
    # (This part remains the same)
    num_boundary_points = 32
    boundary_angles = np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
    boundary_radius = circle_radius * 50
    boundary_points_x = circle_center[0] + boundary_radius * np.cos(boundary_angles)
    boundary_points_y = circle_center[1] + boundary_radius * np.sin(boundary_angles)
    boundary_points = np.vstack([boundary_points_x, boundary_points_y]).T
    if points.ndim == 1: points = points.reshape(-1, 2)
    if boundary_points.ndim == 1: boundary_points = boundary_points.reshape(-1, 2)
    all_points = np.vstack([points, boundary_points])


    # --- 3. Compute Voronoi Diagram ---
    # (This part remains the same)
    try:
        vor = Voronoi(all_points)
    except Exception as e:
        print(f"Error computing Voronoi diagram: {e}")
        return None, None

    # --- 4. Clip Voronoi Regions, Plot, and Add Labels ---
    fig, ax = plt.subplots(figsize=(12, 12)) # Increased size slightly for labels
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Create the circular boundary polygon
    num_circle_vertices = 100
    circle_angles = np.linspace(0, 2 * np.pi, num_circle_vertices)
    circle_poly_points = [
        (circle_center[0] + circle_radius * np.cos(a),
         circle_center[1] + circle_radius * np.sin(a))
        for a in circle_angles
    ]
    circle_polygon = Polygon(circle_poly_points)

    # Create color map
    category_colors = {cat: color for cat, color in zip(categories, generate_distinct_colors(num_categories))}

    plotted_regions = 0
    min_area_for_label = (circle_radius**2 * np.pi) * 0.001 # Heuristic: Don't label tiny regions (0.1% of circle area)

    for i in range(num_points): # Only process regions for the *original* data points
        region_index = vor.point_region[i]
        if region_index == -1: continue

        vertices_indices = vor.regions[region_index]
        if -1 in vertices_indices:
            vertices_indices = [v for v in vertices_indices if v != -1]
        if not vertices_indices or len(vertices_indices) < 3: continue

        try:
            region_vertices = vor.vertices[vertices_indices]
            # Sometimes Voronoi vertices can be identical, filter them for Shapely
            unique_vertices = []
            for v in region_vertices:
                if not any(np.allclose(v, uv) for uv in unique_vertices):
                    unique_vertices.append(v)
            if len(unique_vertices) < 3: continue # Need at least 3 unique vertices

            voronoi_polygon = Polygon(unique_vertices)
            # Ensure the voronoi polygon itself is valid before clipping
            if not voronoi_polygon.is_valid:
                voronoi_polygon = voronoi_polygon.buffer(0) # Attempt to fix invalid polygon
                if not voronoi_polygon.is_valid or voronoi_polygon.is_empty:
                    print(f"Warning: Skipping invalid Voronoi polygon for point {i} ('{point_data_mapping[i]['name']}')")
                    continue

            clipped_geometry = voronoi_polygon.intersection(circle_polygon)

            if clipped_geometry.is_empty: continue

            # Get data for this point
            item_data = point_data_mapping[i]
            category = item_data['category']
            name = item_data['name']
            color = category_colors[category]

            # Handle both Polygon and MultiPolygon results from intersection
            if isinstance(clipped_geometry, Polygon):
                list_of_polygons = [clipped_geometry]
            elif isinstance(clipped_geometry, MultiPolygon):
                list_of_polygons = list(clipped_geometry.geoms)
            else: # Other geometry types (e.g. LineString) - skip
                 continue

            for poly in list_of_polygons:
                 if isinstance(poly, Polygon) and poly.area > 1e-7: # Check type and avoid tiny slivers
                    poly_verts = np.array(poly.exterior.coords)
                    patch = plt.Polygon(poly_verts, facecolor=color, edgecolor='white', linewidth=0.6)
                    ax.add_patch(patch)
                    plotted_regions += 1 # Count plotted polygons

                    # *** ADD TEXT LABEL ***
                    # Add label only if polygon area is large enough
                    if poly.area > min_area_for_label:
                        centroid = poly.centroid
                        # Choose text color based on background brightness (simple version)
                        bg_lum = np.mean(color[:3]) # Average RGB for luminance approximation
                        text_color = 'white' if bg_lum < 0.5 else 'black'

                        ax.text(centroid.x, centroid.y, name,
                                fontsize=label_fontsize,
                                color=text_color,
                                ha='center', # Horizontal alignment
                                va='center', # Vertical alignment
                                clip_on=True) # Prevent labels from drawing outside axes


        except IndexError:
            # This might happen if vertex indices are out of bounds - less likely with checks
            print(f"Warning: IndexError accessing vertices for region of point {i} ('{item_data['name']}').")
        except Exception as e:
             # Catch other potential errors during polygon processing/plotting
             print(f"Error processing/plotting region for point {i} ('{item_data['name']}'): {e}")


    print(f"Plotted {plotted_regions} polygon regions.")
    if plotted_regions < num_points:
         print(f"Warning: Some points may not have resulted in a visible plotted region after clipping.")

    ax.set_xlim(circle_center[0] - circle_radius * 1.05, circle_center[0] + circle_radius * 1.05)
    ax.set_ylim(circle_center[1] - circle_radius * 1.05, circle_center[1] + circle_radius * 1.05)

    handles = [plt.Rectangle((0,0),1,1, color=category_colors[cat]) for cat in categories]
    ax.legend(handles, categories, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    return fig, ax

# --- Example Usage (same data) ---
example_data = [
    {'name': 'USA', 'weight': 25.4, 'category': 'North America'},
    {'name': 'China', 'weight': 17.9, 'category': 'Asia'},
    {'name': 'Japan', 'weight': 4.2, 'category': 'Asia'},
    {'name': 'Germany', 'weight': 4.0, 'category': 'Europe'},
    {'name': 'India', 'weight': 3.4, 'category': 'Asia'},
    {'name': 'UK', 'weight': 3.0, 'category': 'Europe'},
    {'name': 'France', 'weight': 2.8, 'category': 'Europe'},
    {'name': 'Canada', 'weight': 2.1, 'category': 'North America'},
    {'name': 'Italy', 'weight': 2.0, 'category': 'Europe'},
    {'name': 'Brazil', 'weight': 1.9, 'category': 'South America'},
    {'name': 'Russia', 'weight': 1.8, 'category': 'Europe/Asia'},
    {'name': 'S. Korea', 'weight': 1.7, 'category': 'Asia'}, # Shortened name
    {'name': 'Australia', 'weight': 1.6, 'category': 'Oceania'},
    {'name': 'Mexico', 'weight': 1.4, 'category': 'North America'},
    {'name': 'Spain', 'weight': 1.4, 'category': 'Europe'},
    {'name': 'Indonesia', 'weight': 1.2, 'category': 'Asia'},
    {'name': 'Saudi A.', 'weight': 1.1, 'category': 'Asia'}, # Shortened name
    {'name': 'Netherlands', 'weight': 1.0, 'category': 'Europe'},
    {'name': 'Turkey', 'weight': 0.9, 'category': 'Europe/Asia'},
    {'name': 'Swiss', 'weight': 0.8, 'category': 'Europe'}, # Shortened name
    {'name': 'Argentina', 'weight': 0.6, 'category': 'South America'},
    {'name': 'Nigeria', 'weight': 0.5, 'category': 'Africa'},
    {'name': 'S. Africa', 'weight': 0.4, 'category': 'Africa'}, # Shortened name
    {'name': 'Egypt', 'weight': 0.4, 'category': 'Africa'},
]

# Generate the plot with labels (e.g., font size 7)
fig, ax = create_clustered_circular_voronoi(example_data, circle_radius=10.0, label_fontsize=7)

if fig:
    plt.title("Clustered Circular Voronoi Diagram with Labels")
    plt.show()
else:
    print("Failed to generate plot.")