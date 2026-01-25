"""
Generate base map visualizations of the London Underground network.
"""
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# TfL line colors (official)
LINE_COLORS = {
    'BAK': '#B36305',  # Bakerloo - Brown
    'CEN': '#E32017',  # Central - Red
    'CIR': '#FFD300',  # Circle - Yellow
    'DIS': '#00782A',  # District - Green
    'DLR': '#00A4A7',  # DLR - Teal
    'ELZ': '#6950A1',  # Elizabeth - Purple
    'HAM': '#F3A9BB',  # Hammersmith & City - Pink
    'JUB': '#A0A5A9',  # Jubilee - Grey
    'MET': '#9B0056',  # Metropolitan - Magenta
    'NTN': '#000000',  # Northern - Black
    'PIC': '#003688',  # Piccadilly - Dark Blue
    'TRM': '#84B817',  # Trams - Lime Green
    'VIC': '#0098D4',  # Victoria - Light Blue
    'WAC': '#95CDBA',  # Waterloo & City - Teal
}

LINE_NAMES = {
    'BAK': 'Bakerloo',
    'CEN': 'Central',
    'CIR': 'Circle',
    'DIS': 'District',
    'DLR': 'DLR',
    'ELZ': 'Elizabeth',
    'HAM': 'Hammersmith & City',
    'JUB': 'Jubilee',
    'MET': 'Metropolitan',
    'NTN': 'Northern',
    'PIC': 'Piccadilly',
    'TRM': 'Trams',
    'VIC': 'Victoria',
    'WAC': 'Waterloo & City',
}


def load_stops():
    """Load and process stop points data."""
    stops = feather.read_feather(DATA_DIR / 'ALL-StopPoints.feather')
    stops['Easting'] = pd.to_numeric(stops['Place_Location_Easting'])
    stops['Northing'] = pd.to_numeric(stops['Place_Location_Northing'])

    # Convert to lat/lon using pyproj
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True)
        coords = transformer.transform(stops['Easting'].values, stops['Northing'].values)
        stops['Longitude'] = coords[0]
        stops['Latitude'] = coords[1]
        print("Converted coordinates using pyproj")
    except ImportError:
        print("Warning: pyproj not available, using approximate conversion")
        stops['Latitude'] = 49.0 + (stops['Northing'] / 111000)
        stops['Longitude'] = -7.5 + (stops['Easting'] / (111000 * np.cos(np.radians(stops['Latitude']))))

    return stops


def load_route_links(line_code):
    """Load route links for a specific line."""
    path = DATA_DIR / f'{line_code}-RouteLinks.feather'
    if path.exists():
        return feather.read_feather(path)
    return None


def get_unique_edges(route_links):
    """Extract unique station-to-station edges from route links."""
    return route_links[['From_StopPointRef', 'To_StopPointRef']].drop_duplicates()


def plot_line(ax, edges, stop_coords, color, linewidth=2, alpha=0.8):
    """Plot a single tube line on the axes."""
    for _, edge in edges.iterrows():
        from_stop = edge['From_StopPointRef']
        to_stop = edge['To_StopPointRef']

        if from_stop in stop_coords and to_stop in stop_coords:
            x = [stop_coords[from_stop]['Easting'], stop_coords[to_stop]['Easting']]
            y = [stop_coords[from_stop]['Northing'], stop_coords[to_stop]['Northing']]
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')


def main():
    print("Loading station data...")
    stops = load_stops()
    print(f"Loaded {len(stops)} stations")

    # Create coordinate lookup
    stop_coords = stops.set_index('AtcoCode')[
        ['Easting', 'Northing', 'Latitude', 'Longitude', 'Descriptor_CommonName']
    ].to_dict('index')

    # Load all route links
    print("\nLoading route links...")
    all_edges = {}
    for line_code in LINE_COLORS.keys():
        links = load_route_links(line_code)
        if links is not None:
            edges = get_unique_edges(links)
            all_edges[line_code] = edges
            print(f"  {LINE_NAMES.get(line_code, line_code)}: {len(edges)} edges")

    # ==== Visualization 1: Stations only ====
    print("\nGenerating visualization 1: Stations only...")
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.scatter(stops['Easting'], stops['Northing'],
               c='#1f77b4', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('London Underground Stations\n(Geographic Positions)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_stations_only.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '01_stations_only.png'}")

    # ==== Visualization 2: Full network ====
    print("\nGenerating visualization 2: Full network...")
    fig, ax = plt.subplots(figsize=(14, 16))

    for line_code, edges in all_edges.items():
        plot_line(ax, edges, stop_coords, LINE_COLORS[line_code], linewidth=2.5, alpha=0.8)

    ax.scatter(stops['Easting'], stops['Northing'],
               c='white', s=15, edgecolors='black', linewidth=0.5, zorder=10)

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('London Underground Network\n(Geographic Positions)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    legend_elements = [Line2D([0], [0], color=color, linewidth=3, label=LINE_NAMES.get(code, code))
                       for code, color in LINE_COLORS.items() if code in all_edges]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_full_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '02_full_network.png'}")

    # ==== Visualization 3: Individual lines ====
    print("\nGenerating visualization 3: Individual lines...")
    for line_code, edges in all_edges.items():
        fig, ax = plt.subplots(figsize=(12, 10))

        line_stations = set(edges['From_StopPointRef'].tolist() + edges['To_StopPointRef'].tolist())

        plot_line(ax, edges, stop_coords, LINE_COLORS[line_code], linewidth=3)

        for station in line_stations:
            if station in stop_coords:
                ax.scatter(stop_coords[station]['Easting'], stop_coords[station]['Northing'],
                          c='white', s=50, edgecolors=LINE_COLORS[line_code], linewidth=2, zorder=10)

        ax.set_title(f"{LINE_NAMES.get(line_code, line_code)} Line",
                    fontsize=14, color=LINE_COLORS[line_code], fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'line_{line_code.lower()}.png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  Saved: line_{line_code.lower()}.png")

    # ==== Visualization 4: Incremental build-up ====
    print("\nGenerating visualization 4: Incremental build-up...")
    LINE_ORDER = ['MET', 'DIS', 'CIR', 'NTN', 'BAK', 'PIC', 'CEN', 'VIC', 'JUB', 'HAM', 'WAC', 'DLR', 'ELZ', 'TRM']

    current_edges = {}
    for i, line_code in enumerate(LINE_ORDER):
        if line_code not in all_edges:
            continue

        current_edges[line_code] = all_edges[line_code]

        fig, ax = plt.subplots(figsize=(14, 16))

        for code, edges in current_edges.items():
            plot_line(ax, edges, stop_coords, LINE_COLORS[code], linewidth=2.5)

        all_stations = set()
        for edges in current_edges.values():
            all_stations.update(edges['From_StopPointRef'].tolist())
            all_stations.update(edges['To_StopPointRef'].tolist())

        for station in all_stations:
            if station in stop_coords:
                ax.scatter(stop_coords[station]['Easting'], stop_coords[station]['Northing'],
                          c='white', s=15, edgecolors='black', linewidth=0.5, zorder=10)

        ax.set_title(f'London Underground Network\n(Added: {LINE_NAMES.get(line_code, line_code)} Line)',
                    fontsize=14)
        ax.set_aspect('equal')
        ax.axis('off')

        legend_elements = [Line2D([0], [0], color=LINE_COLORS[c], linewidth=3, label=LINE_NAMES.get(c, c))
                           for c in current_edges.keys()]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'incremental_{i+1:02d}_{line_code}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: incremental_{i+1:02d}_{line_code}.png")

    print(f"\nDone! All visualizations saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
