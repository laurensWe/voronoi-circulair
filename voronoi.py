import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi # Keep Voronoi itself
from shapely.geometry import Polygon, Point
# from shapely.ops import polygonize, unary_union # Not strictly needed for plotting clipped polygons
import random
import colorsys # For generating distinct colors
from collections import defaultdict

def generate_distinct_colors(n):
    """Generates N visually distinct colors."""
    hsv_tuples = [(x * 1.0 / n, 0.8, 0.8) for x in range(n)]
    rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    # Shuffle colors slightly to avoid adjacent similar hues if categories are sorted
    random.shuffle(rgb_tuples)
    return rgb_tuples

def create_clustered_circular_voronoi(data, circle_radius=1.0, circle_center=(0, 0)):
    """
    Generates a circular Voronoi diagram where points are clustered by category
    within angular sectors sized proportionally to total category weight.

    Args:
        data (list): A list of dictionaries, where each dictionary has at least
                     'name' (str), 'weight' (float), and 'category' (str).
                     Weight is used to determine sector size.
        circle_radius (float): The radius of the clipping circle.
        circle_center (tuple): The (x, y) coordinates of the circle's center.

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
    category_weights = defaultdict(float)
    category_counts = defaultdict(int)
    total_weight = 0.0
    for item in data:
        weight = item.get('weight', 1.0) # Default weight is 1 if missing
        if weight < 0:
            print(f"Warning: Item '{item.get('name', '')}' has negative weight {weight}. Using 0 instead for sector calculation.")
            weight = 0.0
        category_weights[item['category']] += weight
        category_counts[item['category']] += 1
        total_weight += weight

    if total_weight <= 0:
        print("Total weight of categories is zero or negative. Cannot assign sectors proportionally.")
        # Fallback: assign equal angles if counts > 0
        if num_points > 0:
             print("Falling back to equal angular sectors based on category counts.")
             total_angle_alloc = sum(category_counts.values()) # should equal num_points
             if total_angle_alloc > 0:
                 category_angles = {}
                 current_angle = 0
                 for category in categories:
                     proportion = category_counts[category] / total_angle_alloc
                     angle_size = proportion * 2 * np.pi
                     category_angles[category] = (current_angle, current_angle + angle_size)
                     current_angle += angle_size
             else: # No points with categories
                 print("Error: No data points with categories found.")
                 return None, None
        else:
            print("Error: No data points provided.")
            return None, None

    else:
        # Assign angles proportionally to weight
        category_angles = {}
        current_angle = 0.0
        # Sort categories for consistent sector ordering (optional, but good practice)
        for category in categories:
            proportion = category_weights[category] / total_weight
            angle_size = proportion * 2 * np.pi
            category_angles[category] = (current_angle, current_angle + angle_size)
            current_angle += angle_size


    # --- 2. Generate Seed Points within Category Sectors ---
    np.random.seed(42) # for reproducibility
    generated_points = []
    point_data_mapping = [] # Store original data associated with each point

    for item in data:
        category = item['category']
        angle_start, angle_end = category_angles[category]

        # Generate random radius (sqrt for uniform area distribution)
        r = np.sqrt(np.random.rand()) * circle_radius
        # Generate random angle within the category's sector
        theta = angle_start + np.random.rand() * (angle_end - angle_start)

        # Convert polar to Cartesian
        x = circle_center[0] + r * np.cos(theta)
        y = circle_center[1] + r * np.sin(theta)

        generated_points.append((x, y))
        point_data_mapping.append(item) # Keep track of which point belongs to which item

    points = np.array(generated_points)

    # --- Add Boundary Points (same as before) ---
    num_boundary_points = 32
    boundary_angles = np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
    boundary_radius = circle_radius * 50 # Far away
    boundary_points_x = circle_center[0] + boundary_radius * np.cos(boundary_angles)
    boundary_points_y = circle_center[1] + boundary_radius * np.sin(boundary_angles)
    boundary_points = np.vstack([boundary_points_x, boundary_points_y]).T

    # Ensure points and boundary_points are 2D arrays before vstack
    if points.ndim == 1: points = points.reshape(-1, 2)
    if boundary_points.ndim == 1: boundary_points = boundary_points.reshape(-1, 2)

    all_points = np.vstack([points, boundary_points])

    # --- 3. Compute Voronoi Diagram ---
    try:
        vor = Voronoi(all_points)
    except Exception as e:
        print(f"Error computing Voronoi diagram: {e}")
        print("This might happen with too few points or co-linear/co-circular points.")
        return None, None

    # --- 4. Clip Voronoi Regions and Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
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

    # Create color map (moved here as categories are finalized now)
    category_colors = {cat: color for cat, color in zip(categories, generate_distinct_colors(num_categories))}


    plotted_regions = 0
    for i in range(num_points): # Only process regions for the *original* data points
        region_index = vor.point_region[i]
        if region_index == -1:
            # This can happen if points are *very* close, especially on sector boundaries
            print(f"Warning: Point {i} ('{point_data_mapping[i]['name']}') could not be assigned a Voronoi region.")
            continue

        vertices_indices = vor.regions[region_index]

        if -1 in vertices_indices:
            vertices_indices = [v for v in vertices_indices if v != -1]
            if len(vertices_indices) < 3:
                 # print(f"Warning: Infinite region for point {i} ('{point_data_mapping[i]['name']}') resulted in < 3 vertices.")
                 continue

        if not vertices_indices or len(vertices_indices) < 3:
             # print(f"Warning: Region for point {i} ('{point_data_mapping[i]['name']}') has < 3 vertices.")
            continue

        try:
            region_vertices = vor.vertices[vertices_indices]
            voronoi_polygon = Polygon(region_vertices)
            clipped_polygon = voronoi_polygon.intersection(circle_polygon)

            if clipped_polygon.is_empty:
                continue

            # Get category from the mapping we created during point generation
            category = point_data_mapping[i]['category']
            color = category_colors[category]

            if isinstance(clipped_polygon, Polygon):
                poly_to_plot = [clipped_polygon]
            else: # MultiPolygon
                poly_to_plot = list(clipped_polygon.geoms)

            for poly in poly_to_plot:
                if poly.area > 1e-7: # Avoid plotting tiny slivers
                    poly_verts = np.array(poly.exterior.coords)
                    # Use slightly thicker lines maybe for visual separation
                    patch = plt.Polygon(poly_verts, facecolor=color, edgecolor='white', linewidth=0.8)
                    ax.add_patch(patch)
                    plotted_regions += 1

        except IndexError:
            print(f"Warning: IndexError accessing vertices for region of point {i} ('{point_data_mapping[i]['name']}'). Indices: {vertices_indices}")
        except Exception as e:
             print(f"Error processing/plotting region for point {i} ('{point_data_mapping[i]['name']}'): {e}")

    print(f"Plotted {plotted_regions} regions.")
    if plotted_regions < num_points:
         print(f"Warning: Expected {num_points} regions based on input data, but only {plotted_regions} were plotted successfully after clipping.")

    ax.set_xlim(circle_center[0] - circle_radius * 1.05, circle_center[0] + circle_radius * 1.05)
    ax.set_ylim(circle_center[1] - circle_radius * 1.05, circle_center[1] + circle_radius * 1.05)

    handles = [plt.Rectangle((0,0),1,1, color=category_colors[cat]) for cat in categories]
    ax.legend(handles, categories, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    return fig, ax

# --- Example Usage (same data as before) ---
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
    {'name': 'Russia', 'weight': 1.8, 'category': 'Europe/Asia'}, # Could be its own category
    {'name': 'South Korea', 'weight': 1.7, 'category': 'Asia'},
    {'name': 'Australia', 'weight': 1.6, 'category': 'Oceania'},
    {'name': 'Mexico', 'weight': 1.4, 'category': 'North America'},
    {'name': 'Spain', 'weight': 1.4, 'category': 'Europe'},
    {'name': 'Indonesia', 'weight': 1.2, 'category': 'Asia'},
    {'name': 'Saudi Arabia', 'weight': 1.1, 'category': 'Asia'},
    {'name': 'Netherlands', 'weight': 1.0, 'category': 'Europe'},
    {'name': 'Turkey', 'weight': 0.9, 'category': 'Europe/Asia'},
    {'name': 'Switzerland', 'weight': 0.8, 'category': 'Europe'},
    {'name': 'Brazil' , 'weight': 1.9 , 'category': 'South America'} ,
    {'name': 'Argentina', 'weight': 0.6, 'category': 'South America'},
    {'name': 'Nigeria', 'weight': 0.5, 'category': 'Africa'},
    {'name': 'South Africa', 'weight': 0.4, 'category': 'Africa'},
    {'name': 'Egypt', 'weight': 0.4, 'category': 'Africa'},
    # Add more data points as needed
]

# Generate the plot
fig, ax = create_clustered_circular_voronoi(example_data, circle_radius=10.0)

if fig:
    plt.title("Clustered Circular Voronoi Diagram (by Category Weight)")
    plt.show()
else:
    print("Failed to generate plot.")