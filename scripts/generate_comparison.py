"""
Generate comparison view: Geographic vs Schematic tube map.

The schematic is generated algorithmically using force-directed layout
to approximate the classic Beck diagram style.
"""
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import networkx as nx

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# TfL line colors (official)
LINE_COLORS = {
    'BAK': '#B36305',  # Bakerloo
    'CEN': '#E32017',  # Central
    'CIR': '#FFD300',  # Circle
    'DIS': '#00782A',  # District
    'DLR': '#00A4A7',  # DLR
    'ELZ': '#6950A1',  # Elizabeth
    'HAM': '#F3A9BB',  # Hammersmith & City
    'JUB': '#A0A5A9',  # Jubilee
    'MET': '#9B0056',  # Metropolitan
    'NTN': '#000000',  # Northern
    'PIC': '#003688',  # Piccadilly
    'TRM': '#84B817',  # Trams
    'VIC': '#0098D4',  # Victoria
    'WAC': '#95CDBA',  # Waterloo & City
}

LINE_NAMES = {
    'BAK': 'Bakerloo', 'CEN': 'Central', 'CIR': 'Circle', 'DIS': 'District',
    'DLR': 'DLR', 'ELZ': 'Elizabeth', 'HAM': 'Hammersmith & City', 'JUB': 'Jubilee',
    'MET': 'Metropolitan', 'NTN': 'Northern', 'PIC': 'Piccadilly', 'TRM': 'Trams',
    'VIC': 'Victoria', 'WAC': 'Waterloo & City',
}


def load_data():
    """Load stops and route links."""
    # Load stops
    stops = feather.read_feather(DATA_DIR / 'ALL-StopPoints.feather')
    stops['Easting'] = pd.to_numeric(stops['Place_Location_Easting'])
    stops['Northing'] = pd.to_numeric(stops['Place_Location_Northing'])

    # Convert to lat/lon
    from pyproj import Transformer
    transformer = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True)
    coords = transformer.transform(stops['Easting'].values, stops['Northing'].values)
    stops['Longitude'] = coords[0]
    stops['Latitude'] = coords[1]

    # Create lookup
    stop_coords = stops.set_index('AtcoCode')[
        ['Easting', 'Northing', 'Latitude', 'Longitude', 'Descriptor_CommonName']
    ].to_dict('index')

    # Load route links for all lines
    all_edges = {}
    for line_code in LINE_COLORS.keys():
        path = DATA_DIR / f'{line_code}-RouteLinks.feather'
        if path.exists():
            links = feather.read_feather(path)
            edges = links[['From_StopPointRef', 'To_StopPointRef']].drop_duplicates()
            all_edges[line_code] = edges

    return stops, stop_coords, all_edges


def build_network_graph(all_edges, stop_coords):
    """Build a NetworkX graph from edges."""
    G = nx.Graph()

    # Add all stations as nodes with geographic positions
    for station_id, data in stop_coords.items():
        G.add_node(station_id,
                   pos_geo=(data['Easting'], data['Northing']),
                   name=data['Descriptor_CommonName'])

    # Add edges with line information
    for line_code, edges in all_edges.items():
        for _, edge in edges.iterrows():
            from_s, to_s = edge['From_StopPointRef'], edge['To_StopPointRef']
            if from_s in stop_coords and to_s in stop_coords:
                if G.has_edge(from_s, to_s):
                    G[from_s][to_s]['lines'].append(line_code)
                else:
                    G.add_edge(from_s, to_s, lines=[line_code])

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G


def generate_schematic_positions(G, iterations=500):
    """
    Generate schematic positions using force-directed layout
    with modifications to encourage horizontal/vertical/diagonal lines.
    """
    # Get geographic positions as starting point
    pos_geo = nx.get_node_attributes(G, 'pos_geo')

    # Normalize geographic positions to [0, 1] range
    if not pos_geo:
        return {}

    xs = [p[0] for p in pos_geo.values()]
    ys = [p[1] for p in pos_geo.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    pos_init = {
        node: ((p[0] - x_min) / x_range, (p[1] - y_min) / y_range)
        for node, p in pos_geo.items()
    }

    # Use spring layout with geographic positions as seed
    # Higher k = more spread out, iterations for convergence
    pos_schematic = nx.spring_layout(
        G,
        pos=pos_init,
        k=2.0 / np.sqrt(len(G.nodes())),  # Optimal distance
        iterations=iterations,
        seed=42
    )

    # Scale back to similar range as geographic
    for node in pos_schematic:
        x, y = pos_schematic[node]
        pos_schematic[node] = (
            x * x_range + x_min,
            y * y_range + y_min
        )

    return pos_schematic


def snap_to_octagonal(pos_schematic, G, strength=0.3):
    """
    Adjust edge directions toward 45-degree increments (octagonal grid).
    This creates a more Beck-like schematic appearance.
    """
    pos = {k: list(v) for k, v in pos_schematic.items()}  # Make mutable

    for _ in range(50):  # Iterate to converge
        adjustments = {node: [0.0, 0.0] for node in pos}

        for u, v in G.edges():
            if u not in pos or v not in pos:
                continue

            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]

            if dx == 0 and dy == 0:
                continue

            # Current angle
            angle = np.arctan2(dy, dx)

            # Snap to nearest 45-degree increment
            snapped_angle = round(angle / (np.pi / 4)) * (np.pi / 4)

            # Calculate adjustment
            length = np.sqrt(dx**2 + dy**2)
            target_dx = length * np.cos(snapped_angle)
            target_dy = length * np.sin(snapped_angle)

            # Midpoint stays fixed, adjust both endpoints
            adj_x = (target_dx - dx) * strength / 2
            adj_y = (target_dy - dy) * strength / 2

            adjustments[u][0] -= adj_x
            adjustments[u][1] -= adj_y
            adjustments[v][0] += adj_x
            adjustments[v][1] += adj_y

        # Apply adjustments
        for node in pos:
            pos[node][0] += adjustments[node][0]
            pos[node][1] += adjustments[node][1]

    return {k: tuple(v) for k, v in pos.items()}


def plot_network(ax, G, positions, all_edges, title, show_labels=False):
    """Plot the network on given axes."""
    # Draw edges by line
    for line_code, edges in all_edges.items():
        color = LINE_COLORS[line_code]
        for _, edge in edges.iterrows():
            from_s, to_s = edge['From_StopPointRef'], edge['To_StopPointRef']
            if from_s in positions and to_s in positions:
                x = [positions[from_s][0], positions[to_s][0]]
                y = [positions[from_s][1], positions[to_s][1]]
                ax.plot(x, y, color=color, linewidth=2.5, alpha=0.8, solid_capstyle='round')

    # Draw stations
    station_x = [positions[n][0] for n in G.nodes() if n in positions]
    station_y = [positions[n][1] for n in G.nodes() if n in positions]
    ax.scatter(station_x, station_y, c='white', s=20, edgecolors='black', linewidth=0.5, zorder=10)

    # Highlight interchange stations (degree > 2)
    interchanges = [n for n in G.nodes() if G.degree(n) > 2 and n in positions]
    if interchanges:
        ix = [positions[n][0] for n in interchanges]
        iy = [positions[n][1] for n in interchanges]
        ax.scatter(ix, iy, c='white', s=60, edgecolors='black', linewidth=1.5, zorder=11)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')


def create_comparison_figure(G, pos_geo, pos_schematic, all_edges):
    """Create side-by-side comparison figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # Geographic view
    plot_network(ax1, G, pos_geo, all_edges, 'Geographic Reality')

    # Schematic view
    plot_network(ax2, G, pos_schematic, all_edges, 'Schematic (Algorithmic)')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=color, linewidth=3, label=LINE_NAMES.get(code, code))
        for code, color in LINE_COLORS.items() if code in all_edges
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('London Underground: Geographic vs Schematic', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    return fig


def main():
    print("Loading data...")
    stops, stop_coords, all_edges = load_data()
    print(f"Loaded {len(stops)} stations, {len(all_edges)} lines")

    print("Building network graph...")
    G = build_network_graph(all_edges, stop_coords)
    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Get geographic positions
    pos_geo = {n: G.nodes[n]['pos_geo'] for n in G.nodes() if 'pos_geo' in G.nodes[n]}

    print("Generating schematic layout...")
    pos_schematic = generate_schematic_positions(G, iterations=300)

    print("Snapping to octagonal grid...")
    pos_schematic = snap_to_octagonal(pos_schematic, G, strength=0.2)

    # Create comparison figure
    print("Creating comparison figure...")
    fig = create_comparison_figure(G, pos_geo, pos_schematic, all_edges)

    output_path = OUTPUT_DIR / 'comparison_geo_vs_schematic.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

    # Also create individual high-res versions
    print("Creating individual figures...")

    # Geographic only
    fig, ax = plt.subplots(figsize=(14, 16))
    plot_network(ax, G, pos_geo, all_edges, 'London Underground\n(Geographic Positions)')
    legend_elements = [Line2D([0], [0], color=color, linewidth=3, label=LINE_NAMES.get(code, code))
                       for code, color in LINE_COLORS.items() if code in all_edges]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    fig.savefig(OUTPUT_DIR / 'geographic_network.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Schematic only
    fig, ax = plt.subplots(figsize=(14, 16))
    plot_network(ax, G, pos_schematic, all_edges, 'London Underground\n(Schematic Layout)')
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    fig.savefig(OUTPUT_DIR / 'schematic_network.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()
