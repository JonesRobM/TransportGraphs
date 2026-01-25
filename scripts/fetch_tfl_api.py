"""
Fetch line data from the TfL Unified API.

This script fetches route and station data for lines not available in
TransXChange format (e.g., Elizabeth line, London Overground).

Usage:
    python fetch_tfl_api.py                    # Fetch Elizabeth line
    python fetch_tfl_api.py elizabeth          # Fetch Elizabeth line
    python fetch_tfl_api.py overground         # Fetch London Overground
    python fetch_tfl_api.py elizabeth overground  # Fetch multiple lines

API Documentation: https://api.tfl.gov.uk/
"""
import requests
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
import sys
import json

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR.mkdir(exist_ok=True)

# TfL API base URL
API_BASE = 'https://api.tfl.gov.uk'

# Line ID mappings (TfL API ID -> our code)
LINE_MAPPINGS = {
    'elizabeth': 'ELZ',
    'dlr': 'DLR',
    'tram': 'TRM',
    # London Overground lines (rebranded 2024)
    'liberty': 'LIB',
    'lioness': 'LIO',
    'mildmay': 'MIL',
    'suffragette': 'SUF',
    'weaver': 'WEA',
    'windrush': 'WIN',
}

# Aliases for fetching multiple lines at once
LINE_ALIASES = {
    'overground': ['liberty', 'lioness', 'mildmay', 'suffragette', 'weaver', 'windrush'],
    'all': ['elizabeth', 'dlr', 'tram', 'liberty', 'lioness', 'mildmay', 'suffragette', 'weaver', 'windrush'],
}


def fetch_line_route(line_id):
    """Fetch route sequence data for a line."""
    url = f'{API_BASE}/Line/{line_id}/Route/Sequence/all'
    print(f"Fetching: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_line_stoppoints(line_id):
    """Fetch all stop points for a line."""
    url = f'{API_BASE}/Line/{line_id}/StopPoints'
    print(f"Fetching: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def latlon_to_osgb(lat, lon):
    """
    Convert WGS84 lat/lon to OSGB36 Easting/Northing.
    Uses pyproj if available, otherwise approximate conversion.
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True)
        easting, northing = transformer.transform(lon, lat)
        return int(easting), int(northing)
    except ImportError:
        # Approximate conversion for London area
        # This is rough - pyproj is more accurate
        northing = int((lat - 49.0) * 111000)
        easting = int((lon + 7.5) * 111000 * 0.6)  # cos(51.5) ≈ 0.6
        return easting, northing


def process_stoppoints(data, stoppoints_data):
    """Extract and process stop points from API response."""
    stops = {}

    # Process from route sequence data
    if 'stopPointSequences' in data:
        for seq in data['stopPointSequences']:
            for sp in seq.get('stopPoint', []):
                stop_id = sp.get('id') or sp.get('stationId')
                if stop_id and stop_id not in stops:
                    lat = sp.get('lat', 0)
                    lon = sp.get('lon', 0)
                    easting, northing = latlon_to_osgb(lat, lon)

                    stops[stop_id] = {
                        'AtcoCode': stop_id,
                        'Descriptor_CommonName': sp.get('name', '').replace(' Underground Station', '').replace(' Rail Station', '').replace(' DLR Station', ''),
                        'Place_NptgLocalityRef': sp.get('icsId', ''),
                        'Place_Location_Easting': str(easting),
                        'Place_Location_Northing': str(northing),
                        'Latitude': lat,
                        'Longitude': lon,
                        'Zone': sp.get('zone', ''),
                    }

    # Also process from stoppoints endpoint for more complete data
    for sp in stoppoints_data:
        stop_id = sp.get('stationNaptan') or sp.get('naptanId') or sp.get('id')
        if stop_id and stop_id not in stops:
            lat = sp.get('lat', 0)
            lon = sp.get('lon', 0)
            easting, northing = latlon_to_osgb(lat, lon)

            stops[stop_id] = {
                'AtcoCode': stop_id,
                'Descriptor_CommonName': sp.get('commonName', '').replace(' Underground Station', '').replace(' Rail Station', '').replace(' DLR Station', ''),
                'Place_NptgLocalityRef': sp.get('icsCode', ''),
                'Place_Location_Easting': str(easting),
                'Place_Location_Northing': str(northing),
                'Latitude': lat,
                'Longitude': lon,
                'Zone': sp.get('zone', ''),
            }

    return list(stops.values())


def process_route_links(data):
    """Extract route links (station-to-station connections) from API response."""
    links = []
    seen = set()

    if 'stopPointSequences' in data:
        for seq in data['stopPointSequences']:
            stop_points = seq.get('stopPoint', [])
            for i in range(len(stop_points) - 1):
                from_sp = stop_points[i]
                to_sp = stop_points[i + 1]

                from_id = from_sp.get('id') or from_sp.get('stationId')
                to_id = to_sp.get('id') or to_sp.get('stationId')

                if from_id and to_id:
                    # Create unique key for deduplication
                    key = (from_id, to_id)
                    if key not in seen:
                        seen.add(key)
                        links.append({
                            'RouteLink_id': f'{from_id}_{to_id}',
                            'From_StopPointRef': from_id,
                            'To_StopPointRef': to_id,
                            'Distance': None,
                            'BranchId': seq.get('branchId', 0),
                        })

    return links


def process_journey_patterns(data):
    """Create journey patterns from route sequences."""
    patterns = []

    if 'stopPointSequences' in data:
        for seq in data['stopPointSequences']:
            branch_id = seq.get('branchId', 0)
            stop_points = seq.get('stopPoint', [])

            if len(stop_points) >= 2:
                first_stop = stop_points[0].get('name', 'Unknown')
                last_stop = stop_points[-1].get('name', 'Unknown')

                patterns.append({
                    'JourneyPattern': f'JP_{branch_id}',
                    'DestinationDisplay': last_stop,
                    'Direction': seq.get('direction', 'unknown'),
                    'RouteRef': f'Route_{branch_id}',
                    'JourneyPatternSectionRefs': f'JPS_{branch_id}',
                    'FirstStop': first_stop,
                    'StopCount': len(stop_points),
                })

    return patterns


def process_timing_links(data):
    """Create timing links from route sequences."""
    timing_links = []

    if 'stopPointSequences' in data:
        for seq in data['stopPointSequences']:
            branch_id = seq.get('branchId', 0)
            section_id = f'JPS_{branch_id}'
            stop_points = seq.get('stopPoint', [])

            for i in range(len(stop_points) - 1):
                from_sp = stop_points[i]
                to_sp = stop_points[i + 1]

                from_id = from_sp.get('id') or from_sp.get('stationId')
                to_id = to_sp.get('id') or to_sp.get('stationId')

                if from_id and to_id:
                    timing_links.append({
                        'JourneyPatternSections': section_id,
                        'JourneyPatternTimingLink_id': f'JPTL_{branch_id}_{i}',
                        'From_StopPointRef': from_id,
                        'From_SequenceNumber': i + 1,
                        'To_StopPointRef': to_id,
                        'To_SequenceNumber': i + 2,
                        'RouteLinkRef': f'{from_id}_{to_id}',
                        'RunTime': 'PT2M',  # Default 2 min between stations
                        'WaitTime': 'PT30S',  # Default 30 sec dwell time
                    })

    return timing_links


def process_routes(data):
    """Extract route information."""
    routes = []

    if 'stopPointSequences' in data:
        for seq in data['stopPointSequences']:
            branch_id = seq.get('branchId', 0)
            stop_points = seq.get('stopPoint', [])

            if len(stop_points) >= 2:
                first_stop = stop_points[0].get('name', 'Unknown')
                last_stop = stop_points[-1].get('name', 'Unknown')

                routes.append({
                    'Route_id': f'Route_{branch_id}',
                    'PrivateCode': str(branch_id),
                    'Description': f'{first_stop} to {last_stop}',
                })

    return routes


def process_services(data, line_id, line_code):
    """Create service information."""
    return [{
        'ServiceCode': line_code,
        'PrivateCode': line_id,
        'Mode': data.get('modeName', 'rail'),
        'Description': data.get('lineName', line_id.replace('-', ' ').title()),
    }]


def save_line_data(line_code, data_dict):
    """Save processed data to feather files."""
    for name, records in data_dict.items():
        if records:
            df = pd.DataFrame(records)
            output_path = OUTPUT_DIR / f'{line_code}-{name}.feather'
            feather.write_feather(df, output_path)
            print(f"  Saved: {output_path.name} ({len(df)} records)")


def update_all_stoppoints():
    """Merge all StopPoints into a single file."""
    all_stops = []
    for feather_file in OUTPUT_DIR.glob('*-StopPoints.feather'):
        df = feather.read_feather(feather_file)
        all_stops.append(df)

    if all_stops:
        combined = pd.concat(all_stops, ignore_index=True)
        combined = combined.drop_duplicates(subset=['AtcoCode'])
        output_path = OUTPUT_DIR / 'ALL-StopPoints.feather'
        feather.write_feather(combined, output_path)
        print(f"\nUpdated ALL-StopPoints.feather: {len(combined)} stations")


def process_line(line_id):
    """Fetch and process a single line from the TfL API."""
    line_code = LINE_MAPPINGS.get(line_id, line_id.upper()[:3])
    print(f"\nProcessing {line_id} -> {line_code}")

    try:
        # Fetch data from API
        route_data = fetch_line_route(line_id)
        stoppoints_data = fetch_line_stoppoints(line_id)

        # Process data
        stops = process_stoppoints(route_data, stoppoints_data)
        route_links = process_route_links(route_data)
        patterns = process_journey_patterns(route_data)
        timing_links = process_timing_links(route_data)
        routes = process_routes(route_data)
        services = process_services(route_data, line_id, line_code)

        print(f"  StopPoints: {len(stops)}")
        print(f"  RouteLinks: {len(route_links)}")
        print(f"  JourneyPatterns: {len(patterns)}")
        print(f"  JourneyPatternTimingLinks: {len(timing_links)}")
        print(f"  Routes: {len(routes)}")
        print(f"  Services: {len(services)}")

        # Save data
        save_line_data(line_code, {
            'StopPoints': stops,
            'RouteLinks': route_links,
            'JourneyPatterns': patterns,
            'JourneyPatternTimingLinks': timing_links,
            'Routes': routes,
            'Services': services,
        })

        return True

    except requests.RequestException as e:
        print(f"  Error fetching data: {e}")
        return False


def main():
    # Determine which lines to process
    if len(sys.argv) > 1:
        args = [arg.lower() for arg in sys.argv[1:]]
    else:
        args = ['elizabeth']  # Default to Elizabeth line

    # Expand aliases
    lines = []
    for arg in args:
        if arg in LINE_ALIASES:
            lines.extend(LINE_ALIASES[arg])
            print(f"Expanding '{arg}' to: {', '.join(LINE_ALIASES[arg])}")
        else:
            lines.append(arg)

    success_count = 0
    for line_id in lines:
        if process_line(line_id):
            success_count += 1

    if success_count > 0:
        update_all_stoppoints()

    print(f"\nDone! Processed {success_count}/{len(lines)} lines.")


if __name__ == '__main__':
    main()
